"""
Deployment module for Homeostasis.

This module provides deployment capabilities for the Homeostasis self-healing system,
including canary deployments, blue-green deployments, and rollback mechanisms.
"""

from modules.deployment.blue_green import (BlueGreenDeployment,
                                           BlueGreenStatus, DeploymentColor,
                                           get_blue_green_deployment)
# Import main components
from modules.deployment.canary import (CanaryDeployment, CanaryStatus,
                                       get_canary_deployment)
from modules.deployment.traffic_manager import (KubernetesTrafficManager,
                                                NginxTrafficManager,
                                                TrafficSplitter,
                                                get_kubernetes_manager,
                                                get_nginx_manager,
                                                get_traffic_splitter)

__all__ = [
    # Canary Deployment
    "CanaryDeployment",
    "CanaryStatus",
    "get_canary_deployment",
    # Blue-Green Deployment
    "BlueGreenDeployment",
    "BlueGreenStatus",
    "DeploymentColor",
    "get_blue_green_deployment",
    # Traffic Management
    "TrafficSplitter",
    "NginxTrafficManager",
    "KubernetesTrafficManager",
    "get_traffic_splitter",
    "get_nginx_manager",
    "get_kubernetes_manager",
]
