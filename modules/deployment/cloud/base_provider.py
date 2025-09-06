"""
Base Cloud Provider for Homeostasis.

This module provides a base class for cloud providers.
"""

import abc
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BaseCloudProvider(abc.ABC):
    """
    Base class for cloud providers.

    This class defines the interface that all cloud providers must implement.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize base cloud provider.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.region = self.config.get("region", None)

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if cloud provider is available.

        Returns:
            bool: True if cloud provider is available, False otherwise
        """
        pass

    @abc.abstractmethod
    def deploy_service(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy a service to the cloud provider.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        pass

    @abc.abstractmethod
    def undeploy_service(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy a service from the cloud provider.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        pass

    @abc.abstractmethod
    def get_service_status(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Get status of a deployed service.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Service status information
        """
        pass

    @abc.abstractmethod
    def get_service_logs(
        self, service_name: str, fix_id: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get logs for a deployed service.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters

        Returns:
            List[Dict[str, Any]]: Service logs
        """
        pass
