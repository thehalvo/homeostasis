"""
Resource management for testing.

This module provides utilities for:
1. Managing test timeouts
2. Monitoring and limiting resource usage
3. Tracking resource metrics during tests
"""
import os
import time
import signal
import threading
import psutil
import functools
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent

from modules.monitoring.logger import MonitoringLogger


class ResourceManager:
    """
    Manages resources for test execution.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the resource manager.
        
        Args:
            log_level: Logging level
        """
        self.logger = MonitoringLogger("resource_manager", log_level=log_level)
        self.metrics = {}
        self.running_tests = {}
        
    def with_timeout(self, func: Callable, *args, timeout: int = 30, **kwargs) -> Any:
        """
        Run a function with a timeout.
        
        Args:
            func: Function to run
            *args: Arguments to pass to the function
            timeout: Timeout in seconds
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            TimeoutError: If the function times out
        """
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            self.logger.warning(f"Function {func.__name__} timed out after {timeout} seconds")
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
        
        if exception:
            raise exception
            
        return result
    
    def track_resources(self, test_id: str, interval: float = 1.0) -> None:
        """
        Start tracking resources for a test.
        
        Args:
            test_id: ID of the test
            interval: Interval in seconds between resource checks
        """
        if test_id in self.running_tests:
            self.logger.warning(f"Test {test_id} is already being tracked")
            return
        
        self.logger.info(f"Starting resource tracking for test {test_id}")
        
        # Initialize metrics for this test
        self.metrics[test_id] = {
            "start_time": time.time(),
            "cpu_usage": [],
            "memory_usage": [],
            "elapsed_time": 0
        }
        
        # Create a flag to signal the tracking thread to stop
        stop_flag = threading.Event()
        self.running_tests[test_id] = stop_flag
        
        # Start tracking in a separate thread
        def _track_resources():
            process = psutil.Process()
            while not stop_flag.is_set():
                try:
                    # Get CPU and memory usage
                    cpu_percent = process.cpu_percent(interval=0.1)
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                    
                    # Store metrics
                    self.metrics[test_id]["cpu_usage"].append(cpu_percent)
                    self.metrics[test_id]["memory_usage"].append(memory_mb)
                    self.metrics[test_id]["elapsed_time"] = time.time() - self.metrics[test_id]["start_time"]
                    
                    # Sleep for the specified interval
                    time.sleep(interval)
                    
                except Exception as e:
                    self.logger.exception(e, message=f"Error tracking resources for test {test_id}")
                    time.sleep(interval)
        
        # Start the tracking thread
        thread = threading.Thread(target=_track_resources)
        thread.daemon = True
        thread.start()
    
    def stop_tracking(self, test_id: str) -> Dict[str, Any]:
        """
        Stop tracking resources for a test and return metrics.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Resource metrics
        """
        if test_id not in self.running_tests:
            self.logger.warning(f"Test {test_id} is not being tracked")
            return {}
        
        # Signal the tracking thread to stop
        self.running_tests[test_id].set()
        
        # Calculate summary metrics
        if test_id in self.metrics:
            metrics = self.metrics[test_id]
            
            # Calculate summary statistics
            summary = {
                "elapsed_time": metrics["elapsed_time"],
                "max_memory_mb": max(metrics["memory_usage"]) if metrics["memory_usage"] else 0,
                "avg_memory_mb": sum(metrics["memory_usage"]) / len(metrics["memory_usage"]) if metrics["memory_usage"] else 0,
                "max_cpu_percent": max(metrics["cpu_usage"]) if metrics["cpu_usage"] else 0,
                "avg_cpu_percent": sum(metrics["cpu_usage"]) / len(metrics["cpu_usage"]) if metrics["cpu_usage"] else 0
            }
            
            # Log summary
            self.logger.info(
                f"Resource tracking complete for test {test_id}",
                elapsed_time=f"{summary['elapsed_time']:.2f}s",
                max_memory=f"{summary['max_memory_mb']:.2f}MB",
                avg_memory=f"{summary['avg_memory_mb']:.2f}MB",
                max_cpu=f"{summary['max_cpu_percent']:.2f}%",
                avg_cpu=f"{summary['avg_cpu_percent']:.2f}%"
            )
            
            # Update the metrics with the summary
            metrics["summary"] = summary
            
            # Clean up
            del self.running_tests[test_id]
            
            return metrics
        
        return {}
    
    def enforce_resource_limits(self, resource_limits: Dict[str, str] = None) -> None:
        """
        Enforce resource limits on the current process.
        
        Args:
            resource_limits: Dictionary of resource limits
        """
        if not resource_limits:
            return
            
        try:
            process = psutil.Process()
            
            # Get CPU limit
            if "cpu" in resource_limits:
                cpu_limit = float(resource_limits["cpu"])
                # Note: psutil doesn't support setting CPU limits directly
                # This would typically be done using cgroups or Docker limits
                self.logger.info(f"CPU limits must be enforced at the container level: {cpu_limit} cores")
            
            # Get memory limit
            if "memory" in resource_limits:
                memory_str = resource_limits["memory"]
                # Convert to bytes
                if memory_str.endswith("g"):
                    memory_bytes = int(float(memory_str[:-1]) * 1024 * 1024 * 1024)
                elif memory_str.endswith("m"):
                    memory_bytes = int(float(memory_str[:-1]) * 1024 * 1024)
                elif memory_str.endswith("k"):
                    memory_bytes = int(float(memory_str[:-1]) * 1024)
                else:
                    memory_bytes = int(memory_str)
                
                # Set soft limit
                # Note: This isn't reliable on all platforms
                # Docker/cgroups limits are more reliable
                self.logger.info(f"Memory limits should be enforced at the container level: {memory_bytes} bytes")
                
        except Exception as e:
            self.logger.exception(e, message="Failed to enforce resource limits")
    
    def with_resource_limits(self, func: Callable, *args, 
                           resource_limits: Dict[str, str] = None, 
                           timeout: int = 30,
                           **kwargs) -> Any:
        """
        Run a function with resource limits and timeout.
        
        Args:
            func: Function to run
            *args: Arguments to pass to the function
            resource_limits: Dictionary of resource limits
            timeout: Timeout in seconds
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            TimeoutError: If the function times out
        """
        # Enforce resource limits
        self.enforce_resource_limits(resource_limits)
        
        # Generate a unique ID for this run
        import uuid
        test_id = str(uuid.uuid4())
        
        # Start tracking resources
        self.track_resources(test_id)
        
        try:
            # Run the function with timeout
            result = self.with_timeout(func, *args, timeout=timeout, **kwargs)
            return result
        finally:
            # Stop tracking resources
            metrics = self.stop_tracking(test_id)
            
            # Store the metrics for later analysis
            # In a real implementation, you might save these to a database
            self.logger.debug(f"Test {test_id} resource metrics", metrics=metrics)


def timeout(seconds: int):
    """
    Decorator to apply a timeout to a function.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            
            def target():
                nonlocal result
                result = func(*args, **kwargs)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            return result
        
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # Example usage
    manager = ResourceManager()
    
    # Run a function with timeout
    @timeout(5)
    def slow_function():
        time.sleep(10)
        return "Done"
    
    try:
        result = slow_function()
        print(result)
    except TimeoutError as e:
        print(f"Caught timeout: {e}")
    
    # Run a function with resource limits
    def memory_intensive_function():
        # Allocate 100MB
        data = [0] * (25 * 1024 * 1024)  # ~100MB
        time.sleep(2)
        return len(data)
    
    try:
        result = manager.with_resource_limits(
            memory_intensive_function,
            resource_limits={"memory": "50m"},  # 50MB limit
            timeout=5
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Caught exception: {e}")