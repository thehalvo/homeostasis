"""
Universal Self-Healing Standard (USHS) v1.0 - Industry Adoption Module

This module provides integration adapters for various industry platforms
to enable USHS compliance and interoperability.
"""

from typing import Dict, Any, List, Optional

__version__ = "1.0.0"

# Import serverless adapters
from standards.v1.0.industry-adoption.serverless_adapter import (
    ServerlessUSHSAdapter,
    AWSLambdaUSHSAdapter,
    AzureFunctionsUSHSAdapter,
    GCPFunctionsUSHSAdapter,
    VercelUSHSAdapter,
    NetlifyUSHSAdapter,
    CloudflareWorkersUSHSAdapter
)

# Import container orchestration adapters
from standards.v1.0.industry-adoption.container_orchestration_adapter import (
    ContainerOrchestrationUSHSAdapter,
    KubernetesUSHSAdapter,
    DockerSwarmUSHSAdapter,
    NomadUSHSAdapter,
    ECSUSHSAdapter,
    AKSUSHSAdapter,
    GKEUSHSAdapter
)

# Import service mesh adapters
from standards.v1.0.industry-adoption.service_mesh_adapter import (
    ServiceMeshUSHSAdapter,
    IstioUSHSAdapter,
    LinkerdUSHSAdapter,
    ConsulConnectUSHSAdapter,
    AWSAppMeshUSHSAdapter,
    KumaUSHSAdapter
)

# Import edge computing adapters
from standards.v1.0.industry-adoption.edge_computing_adapter import (
    EdgeComputingUSHSAdapter,
    CloudflareEdgeUSHSAdapter,
    FastlyEdgeUSHSAdapter,
    AWSOutpostsUSHSAdapter,
    AzureStackEdgeUSHSAdapter,
    K3sEdgeUSHSAdapter
)

__all__ = [
    # Serverless
    'ServerlessUSHSAdapter',
    'AWSLambdaUSHSAdapter',
    'AzureFunctionsUSHSAdapter',
    'GCPFunctionsUSHSAdapter',
    'VercelUSHSAdapter',
    'NetlifyUSHSAdapter',
    'CloudflareWorkersUSHSAdapter',
    
    # Container Orchestration
    'ContainerOrchestrationUSHSAdapter',
    'KubernetesUSHSAdapter',
    'DockerSwarmUSHSAdapter',
    'NomadUSHSAdapter',
    'ECSUSHSAdapter',
    'AKSUSHSAdapter',
    'GKEUSHSAdapter',
    
    # Service Mesh
    'ServiceMeshUSHSAdapter',
    'IstioUSHSAdapter',
    'LinkerdUSHSAdapter',
    'ConsulConnectUSHSAdapter',
    'AWSAppMeshUSHSAdapter',
    'KumaUSHSAdapter',
    
    # Edge Computing
    'EdgeComputingUSHSAdapter',
    'CloudflareEdgeUSHSAdapter',
    'FastlyEdgeUSHSAdapter',
    'AWSOutpostsUSHSAdapter',
    'AzureStackEdgeUSHSAdapter',
    'K3sEdgeUSHSAdapter'
]


class IndustryAdoptionRegistry:
    """Registry for industry platform adapters."""
    
    def __init__(self):
        self._adapters: Dict[str, Any] = {}
        self._categories: Dict[str, List[str]] = {
            'serverless': [],
            'container_orchestration': [],
            'service_mesh': [],
            'edge_computing': []
        }
    
    def register_adapter(self, category: str, name: str, adapter_class: Any) -> None:
        """Register an industry platform adapter.
        
        Args:
            category: Platform category (serverless, container_orchestration, etc.)
            name: Unique adapter name
            adapter_class: Adapter class implementing USHS interfaces
        """
        if category not in self._categories:
            raise ValueError(f"Unknown category: {category}")
        
        self._adapters[name] = adapter_class
        self._categories[category].append(name)
    
    def get_adapter(self, name: str) -> Optional[Any]:
        """Get an adapter by name.
        
        Args:
            name: Adapter name
            
        Returns:
            Adapter class or None if not found
        """
        return self._adapters.get(name)
    
    def list_adapters(self, category: Optional[str] = None) -> List[str]:
        """List registered adapters.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of adapter names
        """
        if category:
            return self._categories.get(category, [])
        return list(self._adapters.keys())
    
    def get_adapter_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about an adapter.
        
        Args:
            name: Adapter name
            
        Returns:
            Adapter information dictionary
        """
        adapter_class = self._adapters.get(name)
        if not adapter_class:
            return {}
        
        return {
            'name': name,
            'class': adapter_class.__name__,
            'module': adapter_class.__module__,
            'compliant_interfaces': getattr(adapter_class, 'USHS_INTERFACES', []),
            'certification_level': getattr(adapter_class, 'CERTIFICATION_LEVEL', 'Bronze'),
            'supported_languages': getattr(adapter_class, 'SUPPORTED_LANGUAGES', []),
            'supported_features': getattr(adapter_class, 'SUPPORTED_FEATURES', [])
        }


# Global registry instance
registry = IndustryAdoptionRegistry()