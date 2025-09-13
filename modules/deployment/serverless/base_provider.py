"""
Base Serverless Provider for Homeostasis.

This module provides a base class for serverless providers.
"""

import abc
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ServerlessProvider(abc.ABC):
    """
    Base class for serverless providers.

    This class defines the interface that all serverless providers must implement.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base serverless provider.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.region = self.config.get("region", None)

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if serverless provider is available.

        Returns:
            bool: True if serverless provider is available, False otherwise
        """
        pass

    @abc.abstractmethod
    def deploy_function(
        self,
        function_name: str,
        fix_id: str,
        source_path: str,
        handler: str,
        runtime: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy a serverless function.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            source_path: Path to the source code
            handler: Function handler (e.g., "index.handler")
            runtime: Runtime environment (e.g., "python3.9")
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        pass

    @abc.abstractmethod
    def update_function(
        self, function_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Update a serverless function.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Update information
        """
        pass

    @abc.abstractmethod
    def delete_function(
        self, function_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Delete a serverless function.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Deletion information
        """
        pass

    @abc.abstractmethod
    def get_function_status(
        self, function_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Get status of a deployed function.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Function status information
        """
        pass

    @abc.abstractmethod
    def get_function_logs(
        self, function_name: str, fix_id: str, since: Optional[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get logs for a deployed function.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            since: Optional timestamp to get logs since (ISO format)
            **kwargs: Additional provider-specific parameters

        Returns:
            List[Dict[str, Any]]: Function logs
        """
        pass

    @abc.abstractmethod
    def invoke_function(
        self, function_name: str, fix_id: str, payload: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Invoke a serverless function.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            payload: Function payload
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Function response
        """
        pass

    @abc.abstractmethod
    def setup_canary_deployment(
        self, function_name: str, fix_id: str, traffic_percentage: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """Setup canary deployment for a function.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            traffic_percentage: Percentage of traffic to route to new version
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Canary deployment information
        """
        pass

    @abc.abstractmethod
    def update_canary_percentage(
        self, function_name: str, fix_id: str, traffic_percentage: int, **kwargs
    ) -> Dict[str, Any]:
        """Update canary deployment traffic percentage.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            traffic_percentage: New percentage of traffic to route to new version
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Updated canary deployment information
        """
        pass

    @abc.abstractmethod
    def complete_canary_deployment(
        self, function_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Complete canary deployment by promoting new version to 100%.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Completion information
        """
        pass

    @abc.abstractmethod
    def rollback_canary_deployment(
        self, function_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Rollback canary deployment to previous version.

        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Rollback information
        """
        pass
