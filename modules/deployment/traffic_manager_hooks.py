"""
Traffic Manager Infrastructure Hooks for Homeostasis.

This module provides hooks for traffic managers to connect with real infrastructure,
including Nginx, Kubernetes Ingress, and cloud-specific load balancers.
"""

import logging
import os
import json
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class NginxHook:
    """
    Hook for managing Nginx traffic routing.
    
    Provides functions for updating Nginx configuration to split traffic
    between different service versions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Nginx hook.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.nginx_config_path = self.config.get("nginx_config_path", "/etc/nginx/conf.d")
        self.nginx_template_path = self.config.get("nginx_template_path", "modules/deployment/templates/nginx")
        self.nginx_available = self._check_nginx_available()
        
        # Ensure the config path exists - but handle permission errors gracefully
        try:
            os.makedirs(self.nginx_config_path, exist_ok=True)
            os.makedirs(self.nginx_template_path, exist_ok=True)
        except PermissionError:
            # In test mode or when lacking permissions, use temp directories
            import tempfile
            temp_dir = tempfile.gettempdir()
            self.nginx_config_path = os.path.join(temp_dir, "nginx_conf")
            self.nginx_template_path = os.path.join(temp_dir, "nginx_templates")
            os.makedirs(self.nginx_config_path, exist_ok=True)
            os.makedirs(self.nginx_template_path, exist_ok=True)
        
        # Create default templates if they don't exist
        self._ensure_templates_exist()
        
    def _check_nginx_available(self) -> bool:
        """Check if Nginx is available.
        
        Returns:
            bool: True if Nginx is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "nginx"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            nginx_command_available = result.returncode == 0
            
            # Check if Nginx is running
            if nginx_command_available:
                result = subprocess.run(
                    ["pgrep", "nginx"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=False
                )
                nginx_running = result.returncode == 0
                
                if not nginx_running:
                    logger.warning("Nginx is installed but not running")
                    
                return nginx_running
                
            return False
            
        except Exception:
            return False
            
    def _ensure_templates_exist(self) -> None:
        """Create default templates if they don't exist."""
        templates = {
            "upstream.conf.template": """
upstream {service_name} {{
{upstream_servers}
}}
""",
            "server.conf.template": """
server {{
    listen 80;
    server_name {domain};

    location / {{
        proxy_pass http://{service_name};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""
        }
        
        for name, content in templates.items():
            path = os.path.join(self.nginx_template_path, name)
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(content)
                    
    def _reload_nginx(self) -> bool:
        """Reload Nginx configuration.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.nginx_available:
            logger.info("Simulating Nginx reload")
            return True
            
        try:
            result = subprocess.run(
                ["nginx", "-s", "reload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to reload Nginx: {result.stderr.decode()}")
                return False
                
            logger.info("Nginx reloaded successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error reloading Nginx: {str(e)}")
            return False
            
    def _test_nginx_config(self) -> bool:
        """Test Nginx configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if not self.nginx_available:
            logger.info("Simulating Nginx config test")
            return True
            
        try:
            result = subprocess.run(
                ["nginx", "-t"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Nginx configuration test failed: {result.stderr.decode()}")
                return False
                
            logger.info("Nginx configuration test passed")
            return True
            
        except Exception as e:
            logger.exception(f"Error testing Nginx configuration: {str(e)}")
            return False
            
    def update_upstreams(self, service_name: str, 
                       upstreams: Dict[str, Union[int, float]]) -> bool:
        """Update Nginx upstream configuration for a service.
        
        Args:
            service_name: Name of the service
            upstreams: Dictionary of upstream server to weight
                e.g. {"server1:8080": 80, "server2:8080": 20}
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load upstream template
            with open(os.path.join(self.nginx_template_path, "upstream.conf.template"), "r") as f:
                upstream_template = f.read()
                
            # Generate upstream servers configuration
            upstream_servers = []
            for server, weight in upstreams.items():
                upstream_servers.append(f"    server {server} weight={weight};")
                
            # Format template
            upstream_config = upstream_template.format(
                service_name=service_name,
                upstream_servers="\n".join(upstream_servers)
            )
            
            # Write upstream configuration
            upstream_file = os.path.join(self.nginx_config_path, f"{service_name}_upstream.conf")
            with open(upstream_file, "w") as f:
                f.write(upstream_config)
                
            # Test configuration
            if not self._test_nginx_config():
                # Restore previous configuration
                if os.path.exists(f"{upstream_file}.bak"):
                    os.rename(f"{upstream_file}.bak", upstream_file)
                    self._reload_nginx()
                    return False
                    
            # Reload Nginx
            return self._reload_nginx()
            
        except Exception as e:
            logger.exception(f"Error updating Nginx upstreams: {str(e)}")
            return False
            
    def update_server(self, service_name: str, domain: str) -> bool:
        """Update Nginx server configuration for a service.
        
        Args:
            service_name: Name of the service
            domain: Domain name for the service
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load server template
            with open(os.path.join(self.nginx_template_path, "server.conf.template"), "r") as f:
                server_template = f.read()
                
            # Format template
            server_config = server_template.format(
                service_name=service_name,
                domain=domain
            )
            
            # Write server configuration
            server_file = os.path.join(self.nginx_config_path, f"{service_name}_server.conf")
            
            # Backup existing configuration
            if os.path.exists(server_file):
                os.rename(server_file, f"{server_file}.bak")
                
            with open(server_file, "w") as f:
                f.write(server_config)
                
            # Test configuration
            if not self._test_nginx_config():
                # Restore previous configuration
                if os.path.exists(f"{server_file}.bak"):
                    os.rename(f"{server_file}.bak", server_file)
                    self._reload_nginx()
                    return False
                    
            # Reload Nginx
            return self._reload_nginx()
            
        except Exception as e:
            logger.exception(f"Error updating Nginx server: {str(e)}")
            return False
            
    def remove_service(self, service_name: str) -> bool:
        """Remove Nginx configuration for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            upstream_file = os.path.join(self.nginx_config_path, f"{service_name}_upstream.conf")
            server_file = os.path.join(self.nginx_config_path, f"{service_name}_server.conf")
            
            # Backup existing configurations
            if os.path.exists(upstream_file):
                os.rename(upstream_file, f"{upstream_file}.bak")
                
            if os.path.exists(server_file):
                os.rename(server_file, f"{server_file}.bak")
                
            # Reload Nginx
            reload_result = self._reload_nginx()
            
            if not reload_result:
                # Restore previous configurations
                if os.path.exists(f"{upstream_file}.bak"):
                    os.rename(f"{upstream_file}.bak", upstream_file)
                    
                if os.path.exists(f"{server_file}.bak"):
                    os.rename(f"{server_file}.bak", server_file)
                    
                self._reload_nginx()
                return False
                
            # Remove backup files
            if os.path.exists(f"{upstream_file}.bak"):
                os.remove(f"{upstream_file}.bak")
                
            if os.path.exists(f"{server_file}.bak"):
                os.remove(f"{server_file}.bak")
                
            return True
            
        except Exception as e:
            logger.exception(f"Error removing Nginx service: {str(e)}")
            return False


class KubernetesIngressHook:
    """
    Hook for managing Kubernetes Ingress traffic routing.
    
    Provides functions for updating Kubernetes Ingress resources to split traffic
    between different service versions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Kubernetes Ingress hook.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.namespace = self.config.get("namespace", "default")
        self.kubectl_available = self._check_kubectl_available()
        
    def _check_kubectl_available(self) -> bool:
        """Check if kubectl is available.
        
        Returns:
            bool: True if kubectl is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "kubectl"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
            
    def _run_kubectl(self, args: List[str], 
                    input_data: Optional[str] = None) -> Dict[str, Any]:
        """Run kubectl command.
        
        Args:
            args: List of arguments for kubectl
            input_data: Optional input data for kubectl
            
        Returns:
            Dict: Result of kubectl command
        """
        if not self.kubectl_available:
            logger.info(f"Simulating kubectl command: kubectl {' '.join(args)}")
            return {"success": True, "simulated": True}
            
        try:
            cmd = ["kubectl"] + args
            logger.debug(f"Running kubectl command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""
            
            if process.returncode != 0:
                logger.error(f"kubectl command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
            # Try to parse JSON output if possible
            try:
                if stdout and stdout.strip().startswith("{"):
                    result = json.loads(stdout)
                else:
                    result = {"output": stdout}
            except json.JSONDecodeError:
                result = {"output": stdout}
                
            result["success"] = True
            result["returncode"] = process.returncode
            
            return result
            
        except Exception as e:
            logger.exception(f"Error running kubectl command: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def update_ingress(self, name: str, host: str, 
                     services: Dict[str, Union[int, float]]) -> bool:
        """Update Kubernetes Ingress for traffic splitting.
        
        Args:
            name: Name of the Ingress resource
            host: Host name for the Ingress
            services: Dictionary of service name to weight percentage
                e.g. {"service1": 80, "service2": 20}
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.kubectl_available:
            logger.info(f"Simulating Kubernetes Ingress update for {name}")
            return True
            
        try:
            # Check for Ingress API version
            api_versions_result = self._run_kubectl(["api-versions"])
            
            if not api_versions_result.get("success", False) and not api_versions_result.get("simulated", False):
                return False
                
            api_versions = api_versions_result.get("output", "").split()
            
            # Determine Ingress API version
            ingress_api_version = "networking.k8s.io/v1"
            if "networking.k8s.io/v1" in api_versions:
                ingress_api_version = "networking.k8s.io/v1"
            elif "networking.k8s.io/v1beta1" in api_versions:
                ingress_api_version = "networking.k8s.io/v1beta1"
            elif "extensions/v1beta1" in api_versions:
                ingress_api_version = "extensions/v1beta1"
                
            # For Kubernetes with Istio, we would use VirtualService
            # Check if Istio is available
            istio_available = False
            istio_result = self._run_kubectl(["api-resources", "--api-group=networking.istio.io"])
            
            if istio_result.get("success", False) and not istio_result.get("simulated", False):
                istio_available = "virtualservice" in istio_result.get("output", "").lower()
                
            if istio_available:
                # Generate VirtualService YAML
                vs_yaml = self._generate_virtual_service_yaml(name, host, services)
                
                # Apply VirtualService
                with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp_file:
                    temp_file.write(vs_yaml)
                    vs_path = temp_file.name
                    
                apply_result = self._run_kubectl([
                    "apply", "-f", vs_path,
                    "--namespace", self.namespace
                ])
                
                # Clean up temp file
                os.unlink(vs_path)
                
                return apply_result.get("success", False) or apply_result.get("simulated", False)
                
            else:
                # For regular Kubernetes, we'll use Ingress with annotations
                # Generate Ingress YAML based on API version
                if ingress_api_version == "networking.k8s.io/v1":
                    ingress_yaml = self._generate_ingress_v1_yaml(name, host, services)
                else:
                    ingress_yaml = self._generate_ingress_legacy_yaml(name, host, services, ingress_api_version)
                    
                # Apply Ingress
                with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp_file:
                    temp_file.write(ingress_yaml)
                    ingress_path = temp_file.name
                    
                apply_result = self._run_kubectl([
                    "apply", "-f", ingress_path,
                    "--namespace", self.namespace
                ])
                
                # Clean up temp file
                os.unlink(ingress_path)
                
                return apply_result.get("success", False) or apply_result.get("simulated", False)
                
        except Exception as e:
            logger.exception(f"Error updating Kubernetes Ingress: {str(e)}")
            return False
            
    def _generate_virtual_service_yaml(self, name: str, host: str, 
                                     services: Dict[str, Union[int, float]]) -> str:
        """Generate Istio VirtualService YAML.
        
        Args:
            name: Name of the VirtualService resource
            host: Host name for the VirtualService
            services: Dictionary of service name to weight percentage
            
        Returns:
            str: VirtualService YAML
        """
        # Normalize weights
        total_weight = sum(services.values())
        normalized_services = {}
        
        for service, weight in services.items():
            normalized_services[service] = int((weight / total_weight) * 100)
            
        # Adjust for rounding errors
        remaining = 100 - sum(normalized_services.values())
        if remaining != 0:
            # Add remaining to the highest weight service
            max_service = max(normalized_services, key=normalized_services.get)
            normalized_services[max_service] += remaining
            
        # Generate VirtualService YAML
        routes = []
        for service, weight in normalized_services.items():
            routes.append({
                "destination": {
                    "host": service,
                    "port": {
                        "number": 80
                    }
                },
                "weight": weight
            })
            
        virtual_service = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": name,
                "namespace": self.namespace
            },
            "spec": {
                "hosts": [host],
                "gateways": ["mesh"],
                "http": [{
                    "route": routes
                }]
            }
        }
        
        return json.dumps(virtual_service, indent=2)
        
    def _generate_ingress_v1_yaml(self, name: str, host: str, 
                               services: Dict[str, Union[int, float]]) -> str:
        """Generate Kubernetes Ingress YAML for API version v1.
        
        Args:
            name: Name of the Ingress resource
            host: Host name for the Ingress
            services: Dictionary of service name to weight percentage
            
        Returns:
            str: Ingress YAML
        """
        # Note: Standard Kubernetes Ingress doesn't directly support traffic splitting
        # We'll use annotations for Nginx Ingress Controller
        
        # Get the service with the highest weight
        primary_service = max(services, key=services.get)
        
        # Normalize weights for annotation
        total_weight = sum(services.values())
        normalized_services = {}
        
        for service, weight in services.items():
            normalized_services[service] = int((weight / total_weight) * 100)
            
        # Generate canary annotation
        canary_annotation = {}
        if len(services) > 1:
            canary_services = [s for s in services.keys() if s != primary_service]
            if canary_services:
                canary_service = canary_services[0]
                canary_weight = normalized_services[canary_service]
                
                canary_annotation = {
                    "nginx.ingress.kubernetes.io/canary": "true",
                    "nginx.ingress.kubernetes.io/canary-by-header": "X-Canary",
                    "nginx.ingress.kubernetes.io/canary-by-header-value": "true",
                    "nginx.ingress.kubernetes.io/canary-weight": str(canary_weight)
                }
                
        # Generate Ingress YAML
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx"
                }
            },
            "spec": {
                "rules": [{
                    "host": host,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": primary_service,
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        # Add canary annotations if needed
        if canary_annotation:
            ingress["metadata"]["annotations"].update(canary_annotation)
            
        return json.dumps(ingress, indent=2)
        
    def _generate_ingress_legacy_yaml(self, name: str, host: str, 
                                   services: Dict[str, Union[int, float]],
                                   api_version: str) -> str:
        """Generate Kubernetes Ingress YAML for legacy API versions.
        
        Args:
            name: Name of the Ingress resource
            host: Host name for the Ingress
            services: Dictionary of service name to weight percentage
            api_version: API version for the Ingress
            
        Returns:
            str: Ingress YAML
        """
        # Get the service with the highest weight
        primary_service = max(services, key=services.get)
        
        # Normalize weights for annotation
        total_weight = sum(services.values())
        normalized_services = {}
        
        for service, weight in services.items():
            normalized_services[service] = int((weight / total_weight) * 100)
            
        # Generate canary annotation
        canary_annotation = {}
        if len(services) > 1:
            canary_services = [s for s in services.keys() if s != primary_service]
            if canary_services:
                canary_service = canary_services[0]
                canary_weight = normalized_services[canary_service]
                
                canary_annotation = {
                    "nginx.ingress.kubernetes.io/canary": "true",
                    "nginx.ingress.kubernetes.io/canary-by-header": "X-Canary",
                    "nginx.ingress.kubernetes.io/canary-by-header-value": "true",
                    "nginx.ingress.kubernetes.io/canary-weight": str(canary_weight)
                }
                
        # Generate Ingress YAML
        ingress = {
            "apiVersion": api_version,
            "kind": "Ingress",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx"
                }
            },
            "spec": {
                "rules": [{
                    "host": host,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "backend": {
                                "serviceName": primary_service,
                                "servicePort": 80
                            }
                        }]
                    }
                }]
            }
        }
        
        # Add canary annotations if needed
        if canary_annotation:
            ingress["metadata"]["annotations"].update(canary_annotation)
            
        return json.dumps(ingress, indent=2)
        
    def remove_ingress(self, name: str) -> bool:
        """Remove Kubernetes Ingress.
        
        Args:
            name: Name of the Ingress resource
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.kubectl_available:
            logger.info(f"Simulating Kubernetes Ingress removal for {name}")
            return True
            
        try:
            # Check if Istio is available
            istio_available = False
            istio_result = self._run_kubectl(["api-resources", "--api-group=networking.istio.io"])
            
            if istio_result.get("success", False) and not istio_result.get("simulated", False):
                istio_available = "virtualservice" in istio_result.get("output", "").lower()
                
            if istio_available:
                # Delete VirtualService
                vs_result = self._run_kubectl([
                    "delete", "virtualservice", name,
                    "--namespace", self.namespace
                ])
                
                return vs_result.get("success", False) or vs_result.get("simulated", False)
                
            else:
                # Delete Ingress
                ingress_result = self._run_kubectl([
                    "delete", "ingress", name,
                    "--namespace", self.namespace
                ])
                
                return ingress_result.get("success", False) or ingress_result.get("simulated", False)
                
        except Exception as e:
            logger.exception(f"Error removing Kubernetes Ingress: {str(e)}")
            return False


class CloudLoadBalancerHook:
    """
    Hook for managing cloud provider load balancers.
    
    Provides functions for updating load balancer configuration to split traffic
    between different service versions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize cloud load balancer hook.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.provider = self.config.get("provider", "none").lower()
        
        # Initialize provider-specific client
        self.client = None
        
        if self.provider == "aws":
            self._init_aws_client()
        elif self.provider == "gcp":
            self._init_gcp_client()
        elif self.provider == "azure":
            self._init_azure_client()
            
    def _init_aws_client(self) -> None:
        """Initialize AWS client."""
        try:
            # Check if AWS CLI is available
            result = subprocess.run(
                ["which", "aws"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            
            self.aws_available = result.returncode == 0
            
            if not self.aws_available:
                logger.warning("AWS CLI not found, AWS operations will be simulated")
                
        except Exception as e:
            logger.exception(f"Error initializing AWS client: {str(e)}")
            self.aws_available = False
            
    def _init_gcp_client(self) -> None:
        """Initialize GCP client."""
        try:
            # Check if gcloud CLI is available
            result = subprocess.run(
                ["which", "gcloud"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            
            self.gcp_available = result.returncode == 0
            
            if not self.gcp_available:
                logger.warning("gcloud CLI not found, GCP operations will be simulated")
                
        except Exception as e:
            logger.exception(f"Error initializing GCP client: {str(e)}")
            self.gcp_available = False
            
    def _init_azure_client(self) -> None:
        """Initialize Azure client."""
        try:
            # Check if Azure CLI is available
            result = subprocess.run(
                ["which", "az"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            
            self.azure_available = result.returncode == 0
            
            if not self.azure_available:
                logger.warning("Azure CLI not found, Azure operations will be simulated")
                
        except Exception as e:
            logger.exception(f"Error initializing Azure client: {str(e)}")
            self.azure_available = False
            
    def _run_aws_cli(self, args: List[str], 
                   input_data: Optional[str] = None) -> Dict[str, Any]:
        """Run AWS CLI command.
        
        Args:
            args: AWS CLI arguments
            input_data: Optional input data
            
        Returns:
            Dict[str, Any]: Command result
        """
        if not self.aws_available:
            logger.info(f"Simulating AWS CLI command: aws {' '.join(args)}")
            return {"success": True, "simulated": True}
            
        try:
            cmd = ["aws"] + args
            logger.debug(f"Running AWS CLI command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""
            
            if process.returncode != 0:
                logger.error(f"AWS CLI command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
            # Try to parse JSON output if possible
            try:
                if stdout and stdout.strip():
                    result = json.loads(stdout)
                else:
                    result = {"output": stdout}
            except json.JSONDecodeError:
                result = {"output": stdout}
                
            result["success"] = True
            result["returncode"] = process.returncode
            
            return result
            
        except Exception as e:
            logger.exception(f"Error running AWS CLI command: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _run_gcloud(self, args: List[str], 
                  input_data: Optional[str] = None) -> Dict[str, Any]:
        """Run gcloud CLI command.
        
        Args:
            args: gcloud CLI arguments
            input_data: Optional input data
            
        Returns:
            Dict[str, Any]: Command result
        """
        if not self.gcp_available:
            logger.info(f"Simulating gcloud command: gcloud {' '.join(args)}")
            return {"success": True, "simulated": True}
            
        try:
            cmd = ["gcloud"] + args + ["--format=json"]
            logger.debug(f"Running gcloud command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""
            
            if process.returncode != 0:
                logger.error(f"gcloud command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
            # Try to parse JSON output if possible
            try:
                if stdout and stdout.strip():
                    result = json.loads(stdout)
                else:
                    result = {"output": stdout}
            except json.JSONDecodeError:
                result = {"output": stdout}
                
            result["success"] = True
            result["returncode"] = process.returncode
            
            return result
            
        except Exception as e:
            logger.exception(f"Error running gcloud command: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _run_az(self, args: List[str], 
              input_data: Optional[str] = None) -> Dict[str, Any]:
        """Run Azure CLI command.
        
        Args:
            args: Azure CLI arguments
            input_data: Optional input data
            
        Returns:
            Dict[str, Any]: Command result
        """
        if not self.azure_available:
            logger.info(f"Simulating Azure CLI command: az {' '.join(args)}")
            return {"success": True, "simulated": True}
            
        try:
            cmd = ["az"] + args + ["--output", "json"]
            logger.debug(f"Running Azure CLI command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""
            
            if process.returncode != 0:
                logger.error(f"Azure CLI command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
            # Try to parse JSON output if possible
            try:
                if stdout and stdout.strip():
                    result = json.loads(stdout)
                else:
                    result = {"output": stdout}
            except json.JSONDecodeError:
                result = {"output": stdout}
                
            result["success"] = True
            result["returncode"] = process.returncode
            
            return result
            
        except Exception as e:
            logger.exception(f"Error running Azure CLI command: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def update_aws_target_groups(self, load_balancer_name: str, 
                               target_groups: Dict[str, Union[int, float]]) -> bool:
        """Update AWS ALB/NLB target groups.
        
        Args:
            load_balancer_name: Name of the load balancer
            target_groups: Dictionary of target group ARN to weight percentage
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.aws_available:
            logger.info(f"Simulating AWS target group update for {load_balancer_name}")
            return True
            
        try:
            # Get load balancer ARN
            lb_result = self._run_aws_cli([
                "elbv2", "describe-load-balancers",
                "--names", load_balancer_name
            ])
            
            if not lb_result.get("success", False) and not lb_result.get("simulated", False):
                return False
                
            lb_arn = lb_result.get("LoadBalancers", [{}])[0].get("LoadBalancerArn")
            
            if not lb_arn and not lb_result.get("simulated", False):
                logger.error(f"Load balancer '{load_balancer_name}' not found")
                return False
                
            # Get listeners
            listener_result = self._run_aws_cli([
                "elbv2", "describe-listeners",
                "--load-balancer-arn", lb_arn
            ])
            
            if not listener_result.get("success", False) and not listener_result.get("simulated", False):
                return False
                
            listeners = listener_result.get("Listeners", [])
            
            if not listeners and not listener_result.get("simulated", False):
                logger.error(f"No listeners found for load balancer '{load_balancer_name}'")
                return False
                
            # For each listener, update the default action
            for listener in listeners:
                listener_arn = listener.get("ListenerArn")
                
                # Create action for target groups
                actions = []
                for target_group_arn, weight in target_groups.items():
                    actions.append({
                        "Type": "forward",
                        "TargetGroupArn": target_group_arn,
                        "Weight": weight
                    })
                    
                # Update listener
                update_result = self._run_aws_cli([
                    "elbv2", "modify-listener",
                    "--listener-arn", listener_arn,
                    "--default-actions", json.dumps([{
                        "Type": "forward",
                        "ForwardConfig": {
                            "TargetGroups": [{
                                "TargetGroupArn": tg,
                                "Weight": w
                            } for tg, w in target_groups.items()]
                        }
                    }])
                ])
                
                if not update_result.get("success", False) and not update_result.get("simulated", False):
                    return False
                    
            return True
            
        except Exception as e:
            logger.exception(f"Error updating AWS target groups: {str(e)}")
            return False
            
    def update_gcp_backend_service(self, service_name: str, 
                                 backends: Dict[str, Union[int, float]]) -> bool:
        """Update GCP backend service.
        
        Args:
            service_name: Name of the backend service
            backends: Dictionary of backend name to weight percentage
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.gcp_available:
            logger.info(f"Simulating GCP backend service update for {service_name}")
            return True
            
        try:
            # Get project ID
            project_result = self._run_gcloud(["config", "get-value", "project"])
            
            if not project_result.get("success", False) and not project_result.get("simulated", False):
                return False
                
            project_id = project_result.get("output", "").strip()
            
            if not project_id and not project_result.get("simulated", False):
                logger.error("GCP project ID not configured")
                return False
                
            # Get backend service
            service_result = self._run_gcloud([
                "compute", "backend-services", "describe", service_name,
                "--global"
            ])
            
            if not service_result.get("success", False) and not service_result.get("simulated", False):
                return False
                
            # Extract existing backends
            existing_backends = service_result.get("backends", [])
            
            # Update backend weights
            for backend in existing_backends:
                backend_name = backend.get("name")
                if backend_name in backends:
                    # Update weight
                    weight = backends[backend_name]
                    backend["weight"] = weight
                    
            # Update backend service
            with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as temp_file:
                json.dump(service_result, temp_file)
                service_path = temp_file.name
                
            update_result = self._run_gcloud([
                "compute", "backend-services", "update", service_name,
                "--global",
                "--project", project_id,
                "--source-file", service_path
            ])
            
            # Clean up temp file
            os.unlink(service_path)
            
            return update_result.get("success", False) or update_result.get("simulated", False)
            
        except Exception as e:
            logger.exception(f"Error updating GCP backend service: {str(e)}")
            return False
            
    def update_azure_traffic_manager(self, profile_name: str, 
                                   endpoints: Dict[str, Union[int, float]]) -> bool:
        """Update Azure Traffic Manager.
        
        Args:
            profile_name: Name of the Traffic Manager profile
            endpoints: Dictionary of endpoint name to weight
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.azure_available:
            logger.info(f"Simulating Azure Traffic Manager update for {profile_name}")
            return True
            
        try:
            # Get resource group
            resource_group = self.config.get("resource_group")
            
            if not resource_group:
                logger.error("Azure resource group not configured")
                return False
                
            # Get Traffic Manager profile
            profile_result = self._run_az([
                "network", "traffic-manager", "profile", "show",
                "--name", profile_name,
                "--resource-group", resource_group
            ])
            
            if not profile_result.get("success", False) and not profile_result.get("simulated", False):
                return False
                
            # For each endpoint, update the weight
            for endpoint_name, weight in endpoints.items():
                # Update endpoint
                update_result = self._run_az([
                    "network", "traffic-manager", "endpoint", "update",
                    "--name", endpoint_name,
                    "--profile-name", profile_name,
                    "--resource-group", resource_group,
                    "--weight", str(weight)
                ])
                
                if not update_result.get("success", False) and not update_result.get("simulated", False):
                    return False
                    
            return True
            
        except Exception as e:
            logger.exception(f"Error updating Azure Traffic Manager: {str(e)}")
            return False
            
    def update_traffic_split(self, name: str, 
                          targets: Dict[str, Union[int, float]]) -> bool:
        """Update traffic split based on cloud provider.
        
        Args:
            name: Name of the load balancer or service
            targets: Dictionary of target to weight
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.provider == "aws":
            return self.update_aws_target_groups(name, targets)
        elif self.provider == "gcp":
            return self.update_gcp_backend_service(name, targets)
        elif self.provider == "azure":
            return self.update_azure_traffic_manager(name, targets)
        else:
            logger.warning(f"Unsupported cloud provider: {self.provider}")
            return False