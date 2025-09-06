"""
Azure integration for Homeostasis.

This package provides integration with Microsoft Azure for deploying fixes to:
- Function Apps
- Container Instances
- AKS (Azure Kubernetes Service)
"""

from modules.deployment.cloud.azure.provider import AzureProvider

__all__ = ["AzureProvider"]
