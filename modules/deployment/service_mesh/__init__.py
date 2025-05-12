"""
Service Mesh integration module for Homeostasis.

This module provides integration with service mesh technologies
to enable advanced traffic management, observability, and security features.
"""

from modules.deployment.service_mesh.istio_integration import (
    IstioIntegration,
    get_istio_integration
)

__all__ = [
    'IstioIntegration',
    'get_istio_integration'
]