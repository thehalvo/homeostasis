"""
Kubernetes Manager

Provides high-level Kubernetes management functionality.
"""

from .kubernetes_deployment import KubernetesDeployment

# Create alias for compatibility
KubernetesManager = KubernetesDeployment
