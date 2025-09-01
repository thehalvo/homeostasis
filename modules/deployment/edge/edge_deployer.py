"""
Edge deployment functionality for Homeostasis.

Provides a centralized interface for deploying and managing applications
at the edge (CDN, edge servers, etc.) for improved performance and resilience.
"""

import logging
from typing import Dict, List, Any

from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class EdgeDeployer:
    """
    Manages deployment of applications to edge locations.
    
    This class provides a unified interface for deploying and managing
    applications at edge locations across multiple providers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the edge deployer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default settings
        self.default_provider = self.config.get("default_provider", "cloudflare")
        
        # Load provider configurations
        self.provider_configs = self.config.get("providers", {})
        
        # Initialize providers (lazy-loaded)
        self._providers = {}
        
    def _get_provider(self, provider_name: str = None):
        """Get an edge provider by name.
        
        Args:
            provider_name: Name of the provider, or None for default
            
        Returns:
            The edge provider instance
        """
        # Use default if not specified
        if provider_name is None:
            provider_name = self.default_provider
            
        # If already initialized, return it
        if provider_name in self._providers:
            return self._providers[provider_name]
            
        # Initialize the provider
        provider_config = self.provider_configs.get(provider_name, {})
        
        if provider_name.lower() == "cloudflare":
            from modules.deployment.edge.cloudflare import get_cloudflare_provider
            provider = get_cloudflare_provider(provider_config)
        elif provider_name.lower() == "cdn":
            from modules.deployment.edge.cdn import get_cdn_provider
            provider = get_cdn_provider(provider_config)
        else:
            logger.error(f"Unknown edge provider: {provider_name}")
            return None
            
        # Cache the provider
        self._providers[provider_name] = provider
        return provider
        
    def deploy_to_edge(self, service_name: str, fix_id: str, 
                      source_path: str, options: Dict[str, Any] = None,
                      provider_name: str = None) -> Dict[str, Any]:
        """Deploy a service to the edge.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            options: Provider-specific options
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Deployment information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Deploy to the edge
        options = options or {}
        result = provider.deploy(service_name, fix_id, source_path, **options)
        
        # Log the deployment
        try:
            get_audit_logger().log_event(
                event_type="edge_deployment",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def update_edge_deployment(self, service_name: str, fix_id: str, 
                             source_path: str = None, options: Dict[str, Any] = None,
                             provider_name: str = None) -> Dict[str, Any]:
        """Update an edge deployment.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Optional path to the updated source code
            options: Provider-specific options
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Update information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Update the deployment
        options = options or {}
        result = provider.update(service_name, fix_id, source_path, **options)
        
        # Log the update
        try:
            get_audit_logger().log_event(
                event_type="edge_deployment_updated",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def delete_edge_deployment(self, service_name: str, fix_id: str, 
                             provider_name: str = None) -> Dict[str, Any]:
        """Delete an edge deployment.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Deletion information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Delete the deployment
        result = provider.delete(service_name, fix_id)
        
        # Log the deletion
        try:
            get_audit_logger().log_event(
                event_type="edge_deployment_deleted",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def get_edge_deployment_status(self, service_name: str, fix_id: str, 
                                 provider_name: str = None) -> Dict[str, Any]:
        """Get the status of an edge deployment.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Deployment status
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Get the status
        return provider.get_status(service_name, fix_id)
        
    def purge_edge_cache(self, service_name: str, fix_id: str, 
                        paths: List[str] = None, 
                        provider_name: str = None) -> Dict[str, Any]:
        """Purge the cache for an edge deployment.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            paths: Optional list of paths to purge (defaults to all)
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Purge information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Purge the cache
        result = provider.purge_cache(service_name, fix_id, paths)
        
        # Log the purge
        try:
            get_audit_logger().log_event(
                event_type="edge_cache_purged",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "paths": paths,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def setup_edge_canary(self, service_name: str, fix_id: str, 
                        percentage: int = 10, 
                        provider_name: str = None) -> Dict[str, Any]:
        """Setup canary deployment for an edge service.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            percentage: Percentage of traffic to route to canary
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Canary setup information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Setup canary
        result = provider.setup_canary(service_name, fix_id, percentage)
        
        # Log the canary setup
        try:
            get_audit_logger().log_event(
                event_type="edge_canary_setup",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "percentage": percentage,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def update_edge_canary(self, service_name: str, fix_id: str, 
                         percentage: int, 
                         provider_name: str = None) -> Dict[str, Any]:
        """Update canary deployment for an edge service.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            percentage: New percentage of traffic to route to canary
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Canary update information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Update canary
        result = provider.update_canary(service_name, fix_id, percentage)
        
        # Log the canary update
        try:
            get_audit_logger().log_event(
                event_type="edge_canary_updated",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "percentage": percentage,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def complete_edge_canary(self, service_name: str, fix_id: str, 
                           provider_name: str = None) -> Dict[str, Any]:
        """Complete canary deployment for an edge service.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Canary completion information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Complete canary
        result = provider.complete_canary(service_name, fix_id)
        
        # Log the canary completion
        try:
            get_audit_logger().log_event(
                event_type="edge_canary_completed",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def rollback_edge_canary(self, service_name: str, fix_id: str, 
                           provider_name: str = None) -> Dict[str, Any]:
        """Rollback canary deployment for an edge service.
        
        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            provider_name: Name of the provider to use
            
        Returns:
            Dict[str, Any]: Canary rollback information
        """
        # Get the provider
        provider = self._get_provider(provider_name)
        if not provider:
            error_msg = f"No edge provider available for {provider_name or self.default_provider}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Rollback canary
        result = provider.rollback_canary(service_name, fix_id)
        
        # Log the canary rollback
        try:
            get_audit_logger().log_event(
                event_type="edge_canary_rolled_back",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "provider": provider_name or self.default_provider,
                    "success": result.get("success", False)
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result


# Singleton instance
_edge_deployer = None

def get_edge_deployer(config: Dict[str, Any] = None) -> EdgeDeployer:
    """Get or create the singleton EdgeDeployer instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        EdgeDeployer: Singleton instance
    """
    global _edge_deployer
    if _edge_deployer is None:
        _edge_deployer = EdgeDeployer(config)
    return _edge_deployer