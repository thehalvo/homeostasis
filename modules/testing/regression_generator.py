"""
Regression test generator for creating tests for fixed errors.

This module provides utilities for:
1. Automatically creating regression tests for fixed errors
2. Tracking fix history and generating appropriate tests
3. Integrating with the patch generation system
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.monitoring.logger import MonitoringLogger
from modules.testing.test_generator import TestGenerator


class RegressionTestGenerator:
    """
    Generates regression tests for fixed errors.
    """
    
    def __init__(self, 
                output_dir: Optional[Path] = None, 
                history_file: Optional[Path] = None,
                log_level: str = "INFO",
                test_framework: str = "pytest"):
        """
        Initialize the regression test generator.
        
        Args:
            output_dir: Directory to save generated tests
            history_file: File to track regression test history
            log_level: Logging level
            test_framework: Test framework to use (pytest or unittest)
        """
        self.logger = MonitoringLogger("regression_generator", log_level=log_level)
        
        # Set up directories
        self.output_dir = output_dir or (project_root / "tests" / "generated" / "regression")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # History file for tracking regression tests
        self.history_file = history_file or (self.output_dir / "regression_history.json")
        self.history = self._load_history()
        
        # Initialize test generator
        self.test_generator = TestGenerator(
            output_dir=self.output_dir,
            log_level=log_level,
            test_framework=test_framework
        )
        
        self.logger.info(f"Initialized regression test generator using {test_framework}")
    
    def _load_history(self) -> Dict[str, Any]:
        """
        Load regression test history from file.
        
        Returns:
            Regression test history
        """
        if not self.history_file.exists():
            return {"tests": {}, "patches": {}}
            
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.exception(e, message=f"Failed to load regression test history from {self.history_file}")
            return {"tests": {}, "patches": {}}
    
    def _save_history(self) -> None:
        """Save regression test history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            self.logger.exception(e, message=f"Failed to save regression test history to {self.history_file}")
    
    def _patch_key(self, patch: Dict[str, Any]) -> str:
        """
        Generate a unique key for a patch.
        
        Args:
            patch: Patch details
            
        Returns:
            Unique key
        """
        patch_id = patch.get("patch_id", "unknown")
        file_path = patch.get("file_path", "")
        function_name = patch.get("function_name", "")
        bug_id = patch.get("bug_id", "")
        
        # Create a unique key for the patch
        return f"{patch_id}_{bug_id}_{file_path}_{function_name}"
    
    def generate_regression_test(self, 
                               patch: Dict[str, Any], 
                               error_info: Dict[str, Any] = None) -> Optional[Path]:
        """
        Generate a regression test for a fixed error.
        
        Args:
            patch: Patch details
            error_info: Error information from analysis
            
        Returns:
            Path to the generated test or None if failed
        """
        patch_key = self._patch_key(patch)
        
        # Check if we've already generated a test for this patch
        if patch_key in self.history["patches"]:
            test_path = Path(self.history["patches"][patch_key])
            if test_path.exists():
                self.logger.info(f"Using existing regression test for patch {patch_key}")
                return test_path
        
        # Generate a new test
        self.logger.info(f"Generating regression test for patch {patch_key}")
        
        # Generate the test
        test_path = self.test_generator.generate_test_for_patch(patch, error_info)
        
        if test_path:
            # Update history
            rel_path = test_path.relative_to(project_root)
            self.history["patches"][patch_key] = str(rel_path)
            self.history["tests"][str(rel_path)] = {
                "patch_id": patch.get("patch_id", "unknown"),
                "bug_id": patch.get("bug_id", ""),
                "file_path": patch.get("file_path", ""),
                "function_name": patch.get("function_name", ""),
                "generated_at": time.time()
            }
            self._save_history()
            
            self.logger.info(f"Generated regression test at {test_path}")
            return test_path
        
        self.logger.error(f"Failed to generate regression test for patch {patch_key}")
        return None
    
    def generate_regression_tests_for_patches(self, 
                                            patches: List[Dict[str, Any]], 
                                            error_info: List[Dict[str, Any]] = None) -> List[Path]:
        """
        Generate regression tests for multiple patches.
        
        Args:
            patches: List of patches
            error_info: Optional list of error information
            
        Returns:
            List of paths to generated tests
        """
        generated_tests = []
        
        for i, patch in enumerate(patches):
            # Get corresponding error info if available
            patch_error_info = None
            if error_info and i < len(error_info):
                patch_error_info = error_info[i]
                
            test_path = self.generate_regression_test(patch, patch_error_info)
            if test_path:
                generated_tests.append(test_path)
        
        return generated_tests
    
    def run_regression_tests(self, test_command: str = "pytest") -> Tuple[bool, Dict[str, Any]]:
        """
        Run all regression tests.
        
        Args:
            test_command: Command to run tests
            
        Returns:
            Tuple of (success, results)
        """
        if not self.history["tests"]:
            self.logger.info("No regression tests to run")
            return True, {"success": True, "message": "No regression tests to run"}
        
        # Determine test paths
        test_paths = []
        for test_path in self.history["tests"].keys():
            full_path = project_root / test_path
            test_paths.append(str(full_path))
        
        # Create the command
        if test_command == "pytest":
            cmd = f"pytest {' '.join(test_paths)} -v"
        else:
            cmd = f"python -m unittest {' '.join(test_paths)}"
        
        self.logger.info(f"Running regression tests: {cmd}")
        
        # Run the tests
        import subprocess
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            success = result.returncode == 0
            
            results = {
                "success": success,
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if success:
                self.logger.info("All regression tests passed")
            else:
                self.logger.error("Some regression tests failed", stderr=result.stderr)
                
            return success, results
            
        except Exception as e:
            self.logger.exception(e, message="Failed to run regression tests")
            return False, {"success": False, "error": str(e)}
    
    def get_regression_test_stats(self) -> Dict[str, Any]:
        """
        Get statistics about regression tests.
        
        Returns:
            Statistics about regression tests
        """
        num_tests = len(self.history["tests"])
        num_patches = len(self.history["patches"])
        
        # Get stats by file
        file_stats = {}
        for test_info in self.history["tests"].values():
            file_path = test_info.get("file_path", "unknown")
            if file_path not in file_stats:
                file_stats[file_path] = 0
            file_stats[file_path] += 1
        
        # Get stats by bug type (if available)
        bug_stats = {}
        for test_info in self.history["tests"].values():
            bug_id = test_info.get("bug_id", "unknown")
            if bug_id not in bug_stats:
                bug_stats[bug_id] = 0
            bug_stats[bug_id] += 1
        
        return {
            "num_tests": num_tests,
            "num_patches": num_patches,
            "file_stats": file_stats,
            "bug_stats": bug_stats
        }


if __name__ == "__main__":
    # Example usage
    generator = RegressionTestGenerator()
    
    # Create a sample patch
    patch = {
        "patch_id": "test-regression-1",
        "bug_id": "bug_1",
        "file_path": "services/example_service/app.py",
        "function_name": "get_item",
        "patch_code": "try:\n    return items[item_id]\nexcept KeyError:\n    return None"
    }
    
    # Create error info
    error_info = {
        "error_type": "KeyError",
        "parameters": {
            "dict_name": "items",
            "key_name": "item_id"
        }
    }
    
    # Generate a regression test
    test_path = generator.generate_regression_test(patch, error_info)
    
    print(f"Generated regression test: {test_path}")
    
    # Get stats
    stats = generator.get_regression_test_stats()
    
    print(f"Regression test statistics: {stats}")
    
    # Run regression tests
    success, results = generator.run_regression_tests()
    
    print(f"Regression tests {'passed' if success else 'failed'}")