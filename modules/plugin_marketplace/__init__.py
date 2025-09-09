"""
Plugin Marketplace Module

This module provides the infrastructure for the USHS plugin marketplace,
including plugin discovery, validation, security, and distribution.
"""

from .plugin_discovery import (PluginCapability, PluginDiscovery, PluginInfo,
                               PluginManifest, PluginRegistry, PluginStatus,
                               PluginType, PluginValidator)
from .plugin_security import (PermissionType, PluginSandbox,
                              PluginSecurityManager, PluginSigner,
                              SandboxContext, SecurityLevel,
                              VulnerabilityScanner)

__all__ = [
    # Discovery
    "PluginType",
    "PluginStatus",
    "PluginCapability",
    "PluginManifest",
    "PluginInfo",
    "PluginDiscovery",
    "PluginRegistry",
    "PluginValidator",
    # Security
    "SecurityLevel",
    "PermissionType",
    "PluginSandbox",
    "SandboxContext",
    "PluginSigner",
    "VulnerabilityScanner",
    "PluginSecurityManager",
]

__version__ = "1.0.0"
