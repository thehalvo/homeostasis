"""
Traffic Management for Homeostasis.

Provides utilities for managing traffic routing during canary deployments,
including traffic splitting and service proxying.
"""

import logging
import random
from typing import Dict, Optional

from modules.deployment.traffic_manager_hooks import (
    NginxHook,
    KubernetesIngressHook,
    CloudLoadBalancerHook
)

logger = logging.getLogger(__name__)


class TrafficSplitter:
    """
    Manages traffic splitting for canary deployments.
    
    This class provides an abstract interface for traffic splitting that can be
    implemented by different backend mechanisms (e.g., nginx, Kubernetes, etc.)
    """
    
    def __init__(self, config: Dict = None):
        """Initialize traffic splitter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.split_rules = {}  # {service_name: {deployment_id: percentage}}
        
    def set_split_percentage(self, service_name: str, deployment_id: str, 
                            percentage: int) -> bool:
        """Set the percentage of traffic to route to a specific deployment.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            percentage: Percentage of traffic (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate percentage
        if not 0 <= percentage <= 100:
            logger.error(f"Invalid percentage: {percentage}. Must be between 0 and 100.")
            return False
            
        # Initialize service if not exists
        if service_name not in self.split_rules:
            self.split_rules[service_name] = {}
            
        # Set percentage
        self.split_rules[service_name][deployment_id] = percentage
        
        # Normalize percentages to ensure they sum to 100
        self._normalize_percentages(service_name)
        
        logger.info(f"Set traffic split for {service_name}/{deployment_id} to {percentage}%")
        return True
        
    def _normalize_percentages(self, service_name: str) -> None:
        """Normalize percentages to ensure they sum to 100.
        
        Args:
            service_name: Name of the service
        """
        if service_name not in self.split_rules:
            return
            
        total = sum(self.split_rules[service_name].values())
        
        # If only one deployment or total is 0, set to 100%
        if len(self.split_rules[service_name]) == 1 or total == 0:
            for deployment_id in self.split_rules[service_name]:
                self.split_rules[service_name][deployment_id] = 100
            return
            
        # Scale percentages to sum to 100
        if total != 100:
            scale = 100 / total
            for deployment_id in self.split_rules[service_name]:
                self.split_rules[service_name][deployment_id] = int(
                    self.split_rules[service_name][deployment_id] * scale
                )
                
            # Adjust rounding errors
            remaining = 100 - sum(self.split_rules[service_name].values())
            if remaining != 0:
                # Add remaining to the highest percentage
                max_id = max(
                    self.split_rules[service_name], 
                    key=lambda x: self.split_rules[service_name][x]
                )
                self.split_rules[service_name][max_id] += remaining
                
    def remove_deployment(self, service_name: str, deployment_id: str) -> bool:
        """Remove a deployment from traffic splitting.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            
        Returns:
            bool: True if successful, False otherwise
        """
        if service_name not in self.split_rules:
            logger.warning(f"Service {service_name} not found")
            return False
            
        if deployment_id not in self.split_rules[service_name]:
            logger.warning(f"Deployment {deployment_id} not found for service {service_name}")
            return False
            
        # Remove deployment
        del self.split_rules[service_name][deployment_id]
        
        # Normalize remaining percentages
        self._normalize_percentages(service_name)
        
        logger.info(f"Removed traffic split for {service_name}/{deployment_id}")
        return True
        
    def get_split_rules(self, service_name: Optional[str] = None) -> Dict:
        """Get the current traffic split rules.
        
        Args:
            service_name: Optional service name to filter rules
            
        Returns:
            Dict: Traffic split rules
        """
        if service_name:
            return self.split_rules.get(service_name, {})
        return self.split_rules
        
    def should_route_to_canary(self, service_name: str, deployment_id: str) -> bool:
        """Determine if a request should be routed to the canary based on percentage.
        
        This is a simple implementation that makes random choices based on percentages.
        In a real-world implementation, this would be done at the proxy level.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            
        Returns:
            bool: True if request should go to canary, False for stable
        """
        if service_name not in self.split_rules:
            return False
            
        if deployment_id not in self.split_rules[service_name]:
            return False
            
        percentage = self.split_rules[service_name][deployment_id]
        return random.random() * 100 < percentage


class NginxTrafficManager:
    """
    Manages traffic using Nginx for canary deployments.
    
    Generates and updates Nginx configuration files for traffic splitting.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Nginx traffic manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.nginx_config_path = self.config.get("nginx_config_path", "/etc/nginx/conf.d")
        self.template_path = self.config.get("template_path", "modules/deployment/templates/nginx")
        self.traffic_splitter = TrafficSplitter(config)
        
        # Initialize Nginx hook
        self.nginx_hook = NginxHook(config)
        
    def update_nginx_config(self, service_name: str) -> bool:
        """Update the Nginx configuration for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get traffic split rules
            split_rules = self.traffic_splitter.get_split_rules(service_name)
            
            if not split_rules:
                logger.warning(f"No traffic split rules for service {service_name}")
                return False
                
            # Convert deployment IDs to server addresses
            upstreams = {}
            for deployment_id, percentage in split_rules.items():
                # Format the server address based on deployment ID
                server_address = f"{deployment_id}.{service_name}.svc.cluster.local:8080"
                upstreams[server_address] = percentage
                
            # Update Nginx configuration using the hook
            result = self.nginx_hook.update_upstreams(service_name, upstreams)
            
            # Update server configuration if needed
            domain = self.config.get("domain", f"{service_name}.example.com")
            self.nginx_hook.update_server(service_name, domain)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error updating Nginx configuration: {str(e)}")
            return False
            
    def set_split_percentage(self, service_name: str, deployment_id: str, 
                            percentage: int) -> bool:
        """Set the percentage of traffic to route to a specific deployment.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            percentage: Percentage of traffic (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Update traffic splitter
        if not self.traffic_splitter.set_split_percentage(service_name, deployment_id, percentage):
            return False
            
        # Update Nginx configuration
        return self.update_nginx_config(service_name)
        
    def remove_service(self, service_name: str) -> bool:
        """Remove Nginx configuration for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Remove all split rules for the service
        if service_name in self.traffic_splitter.split_rules:
            self.traffic_splitter.split_rules.pop(service_name)
            
        # Remove Nginx configuration
        return self.nginx_hook.remove_service(service_name)


class KubernetesTrafficManager:
    """
    Manages traffic in Kubernetes for canary deployments.
    
    Uses Kubernetes services and destination rules for traffic splitting.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Kubernetes traffic manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.namespace = self.config.get("namespace", "default")
        self.traffic_splitter = TrafficSplitter(config)
        
        # Initialize Kubernetes hook
        self.kubernetes_hook = KubernetesIngressHook(config)
        
        # Check if kubectl is available
        self.kubectl_available = self.kubernetes_hook._check_kubectl_available()
        if not self.kubectl_available:
            logger.warning("kubectl not found, Kubernetes traffic management will be simulated")
            
    def update_service(self, service_name: str, deployment_id: str, 
                      percentage: int) -> bool:
        """Update Kubernetes service for traffic splitting.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            percentage: Percentage of traffic (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Update traffic splitter
        if not self.traffic_splitter.set_split_percentage(service_name, deployment_id, percentage):
            return False
            
        if not self.kubectl_available:
            logger.info(f"Simulating Kubernetes traffic split: {service_name}/{deployment_id} at {percentage}%")
            return True
            
        try:
            # Get current rules
            split_rules = self.traffic_splitter.get_split_rules(service_name)
            
            # Format service names
            services = {}
            for dep_id, pct in split_rules.items():
                # Format service name
                service_k8s_name = f"{service_name}-{dep_id}"
                services[service_k8s_name] = pct
                
            # Update Kubernetes Ingress
            host = self.config.get("host", f"{service_name}.example.com")
            return self.kubernetes_hook.update_ingress(service_name, host, services)
            
        except Exception as e:
            logger.exception(f"Error updating Kubernetes service: {str(e)}")
            return False
            
    def create_canary_deployment(self, service_name: str, deployment_id: str,
                                image: str, percentage: int) -> bool:
        """Create a canary deployment in Kubernetes.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            image: Docker image for the deployment
            percentage: Initial percentage of traffic
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.kubectl_available:
            logger.info(f"Simulating Kubernetes canary deployment: {service_name}/{deployment_id} with {image}")
            return self.traffic_splitter.set_split_percentage(service_name, deployment_id, percentage)
            
        try:
            # In a real implementation, you would generate and apply Kubernetes resources here
            # For now, just log the changes we would make
            logger.info(f"Would create Kubernetes canary deployment:")
            logger.info(f"  - Service: {service_name}")
            logger.info(f"  - Deployment ID: {deployment_id}")
            logger.info(f"  - Image: {image}")
            logger.info(f"  - Initial traffic: {percentage}%")
            
            # Update traffic split
            self.traffic_splitter.set_split_percentage(service_name, deployment_id, percentage)
            
            # Update Kubernetes Ingress
            return self.update_service(service_name, deployment_id, percentage)
            
        except Exception as e:
            logger.exception(f"Error creating Kubernetes canary deployment: {str(e)}")
            return False
            
    def promote_canary(self, service_name: str, deployment_id: str) -> bool:
        """Promote a canary deployment to the primary deployment.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.kubectl_available:
            logger.info(f"Simulating Kubernetes canary promotion: {service_name}/{deployment_id}")
            return self.traffic_splitter.set_split_percentage(service_name, deployment_id, 100)
            
        try:
            # In a real implementation, you would update Kubernetes resources here
            logger.info(f"Would promote Kubernetes canary deployment {deployment_id} to primary")
            
            # Set traffic to 100%
            self.traffic_splitter.set_split_percentage(service_name, deployment_id, 100)
            
            # Update Kubernetes Ingress
            return self.update_service(service_name, deployment_id, 100)
            
        except Exception as e:
            logger.exception(f"Error promoting Kubernetes canary deployment: {str(e)}")
            return False
            
    def rollback_canary(self, service_name: str, deployment_id: str) -> bool:
        """Roll back a canary deployment.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.kubectl_available:
            logger.info(f"Simulating Kubernetes canary rollback: {service_name}/{deployment_id}")
            return self.traffic_splitter.remove_deployment(service_name, deployment_id)
            
        try:
            # In a real implementation, you would update Kubernetes resources here
            logger.info(f"Would roll back Kubernetes canary deployment {deployment_id}")
            
            # Remove deployment from traffic splitting
            self.traffic_splitter.remove_deployment(service_name, deployment_id)
            
            # Update Kubernetes Ingress
            host = self.config.get("host", f"{service_name}.example.com")
            
            # Get remaining split rules
            split_rules = self.traffic_splitter.get_split_rules(service_name)
            
            # Format service names
            services = {}
            for dep_id, pct in split_rules.items():
                # Format service name
                service_k8s_name = f"{service_name}-{dep_id}"
                services[service_k8s_name] = pct
                
            # Update Kubernetes Ingress
            return self.kubernetes_hook.update_ingress(service_name, host, services)
            
        except Exception as e:
            logger.exception(f"Error rolling back Kubernetes canary deployment: {str(e)}")
            return False
            
    def cleanup(self, service_name: str) -> bool:
        """Clean up Kubernetes resources.
        
        Args:
            service_name: Name of the service
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Remove all split rules for the service
        if service_name in self.traffic_splitter.split_rules:
            self.traffic_splitter.split_rules.pop(service_name)
            
        # Remove Kubernetes Ingress
        return self.kubernetes_hook.remove_ingress(service_name)


class CloudTrafficManager:
    """
    Manages traffic using cloud provider load balancers.
    
    Supports AWS, GCP, and Azure.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize cloud traffic manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.provider = self.config.get("provider", "none").lower()
        self.traffic_splitter = TrafficSplitter(config)
        
        # Initialize cloud load balancer hook
        self.cloud_hook = CloudLoadBalancerHook(config)
        
    def update_load_balancer(self, name: str) -> bool:
        """Update cloud load balancer for a service.
        
        Args:
            name: Name of the load balancer or service
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get traffic split rules for the service
        split_rules = self.traffic_splitter.get_split_rules(name)
        
        if not split_rules:
            logger.warning(f"No traffic split rules for service {name}")
            return False
            
        # Update load balancer using the hook
        return self.cloud_hook.update_traffic_split(name, split_rules)
        
    def set_split_percentage(self, service_name: str, deployment_id: str, 
                            percentage: int) -> bool:
        """Set the percentage of traffic to route to a specific deployment.
        
        Args:
            service_name: Name of the service
            deployment_id: ID of the deployment
            percentage: Percentage of traffic (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Update traffic splitter
        if not self.traffic_splitter.set_split_percentage(service_name, deployment_id, percentage):
            return False
            
        # Update load balancer
        return self.update_load_balancer(service_name)


# Singleton instances
_traffic_splitter = None
_nginx_manager = None
_kubernetes_manager = None
_cloud_manager = None

def get_traffic_splitter(config: Dict = None) -> TrafficSplitter:
    """Get traffic splitter singleton.
    
    Args:
        config: Optional configuration
        
    Returns:
        TrafficSplitter: Singleton instance
    """
    global _traffic_splitter
    if _traffic_splitter is None:
        _traffic_splitter = TrafficSplitter(config)
    return _traffic_splitter

def get_nginx_manager(config: Dict = None) -> NginxTrafficManager:
    """Get Nginx traffic manager singleton.
    
    Args:
        config: Optional configuration
        
    Returns:
        NginxTrafficManager: Singleton instance
    """
    global _nginx_manager
    if _nginx_manager is None:
        _nginx_manager = NginxTrafficManager(config)
    return _nginx_manager

def get_kubernetes_manager(config: Dict = None) -> KubernetesTrafficManager:
    """Get Kubernetes traffic manager singleton.
    
    Args:
        config: Optional configuration
        
    Returns:
        KubernetesTrafficManager: Singleton instance
    """
    global _kubernetes_manager
    if _kubernetes_manager is None:
        _kubernetes_manager = KubernetesTrafficManager(config)
    return _kubernetes_manager

def get_cloud_manager(config: Dict = None) -> CloudTrafficManager:
    """Get cloud traffic manager singleton.
    
    Args:
        config: Optional configuration
        
    Returns:
        CloudTrafficManager: Singleton instance
    """
    global _cloud_manager
    if _cloud_manager is None:
        _cloud_manager = CloudTrafficManager(config)
    return _cloud_manager