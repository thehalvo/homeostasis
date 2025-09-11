"""
Edge deployment module for Homeostasis.

This module provides functionality for deploying and managing applications
at the edge (CDN, edge servers, etc.) for improved performance and resilience.
"""

from modules.deployment.edge.cdn import CDNProvider, get_cdn_provider
from modules.deployment.edge.cloudflare import (
    CloudflareProvider,
    get_cloudflare_provider,
)
from modules.deployment.edge.edge_deployer import EdgeDeployer, get_edge_deployer

__all__ = [
    "EdgeDeployer",
    "CloudflareProvider",
    "CDNProvider",
    "get_edge_deployer",
    "get_cloudflare_provider",
    "get_cdn_provider",
]
