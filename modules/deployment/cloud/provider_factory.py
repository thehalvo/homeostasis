"""
Cloud Provider Factory for Homeostasis.

This module provides a factory for creating cloud provider instances.
"""

import logging
from typing import Dict, Optional, Any, Union

from modules.deployment.cloud.base_provider import BaseCloudProvider

logger = logging.getLogger(__name__)


class CloudProviderFactory:
    """
    Factory for creating cloud provider instances.
    
    Supports AWS, GCP, and Azure cloud providers. Provides a unified interface
    for working with different cloud providers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize cloud provider factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.provider_name = self.config.get("provider", "none").lower()
        self._provider = None
        
    def get_provider(self) -> Optional[BaseCloudProvider]:
        """Get cloud provider instance based on configuration.
        
        Returns:
            Optional[BaseCloudProvider]: Cloud provider instance, or None if no provider
                is configured or not supported
        """
        if self._provider is not None:
            return self._provider
            
        if self.provider_name == "none" or not self.provider_name:
            logger.info("No cloud provider configured")
            return None
            
        if self.provider_name == "aws":
            from modules.deployment.cloud.aws.provider import AWSProvider
            self._provider = AWSProvider(self.config.get("aws", {}))
            logger.info("Using AWS cloud provider")
            return self._provider
            
        elif self.provider_name == "gcp":
            from modules.deployment.cloud.gcp.provider import GCPProvider
            self._provider = GCPProvider(self.config.get("gcp", {}))
            logger.info("Using GCP cloud provider")
            return self._provider
            
        elif self.provider_name == "azure":
            from modules.deployment.cloud.azure.provider import AzureProvider
            self._provider = AzureProvider(self.config.get("azure", {}))
            logger.info("Using Azure cloud provider")
            return self._provider
            
        else:
            logger.warning(f"Unsupported cloud provider: {self.provider_name}")
            return None


# Singleton instance for app-wide use
_cloud_provider_factory = None

def get_cloud_provider(config: Dict[str, Any] = None) -> Optional[BaseCloudProvider]:
    """Get or create cloud provider instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Optional[BaseCloudProvider]: Cloud provider instance, or None if no provider
            is configured or not supported
    """
    global _cloud_provider_factory
    
    if _cloud_provider_factory is None:
        _cloud_provider_factory = CloudProviderFactory(config)
        
    return _cloud_provider_factory.get_provider()