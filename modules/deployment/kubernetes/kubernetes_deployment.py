"""
Kubernetes deployment implementation for Homeostasis.

Provides functionality for managing Kubernetes deployments, including
creating, updating, and deleting deployments, services, and ingress resources.
"""

import logging
import os
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)

class KubernetesDeployment:
    """
    Manages Kubernetes deployments for Homeostasis.
    
    Provides methods for creating, updating, and deleting Kubernetes resources,
    including deployments, services, and ingress resources.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Kubernetes deployment manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Set default values from config
        self.namespace = self.config.get("namespace", "default")
        self.deployment_strategy = self.config.get("deployment_strategy", "rolling-update")
        self.service_account = self.config.get("service_account", "default")
        self.resource_limits = self.config.get("resource_limits", {
            "cpu": "1.0",
            "memory": "1g"
        })
        self.health_probes = self.config.get("health_probes", {
            "liveness": {
                "path": "/health",
                "port": 8000,
                "initial_delay": 10,
                "period": 30
            },
            "readiness": {
                "path": "/health",
                "port": 8000,
                "initial_delay": 5,
                "period": 10
            }
        })
        
        # Check if kubectl is available
        self.kubectl_available = self._check_kubectl_available()
        if not self.kubectl_available:
            logger.warning("kubectl not found, Kubernetes operations will be simulated")
            
        # Template directory
        self.template_dir = Path(self.config.get("template_dir", 
                                              "modules/deployment/kubernetes/templates"))
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Create templates if they don't exist
        self._ensure_templates_exist()
        
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
            
    def _ensure_templates_exist(self) -> None:
        """Create default templates if they don't exist."""
        templates = {
            "deployment.yaml": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: {name}
    fix_id: {fix_id}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
      fix_id: {fix_id}
  strategy:
    type: {strategy_type}
  template:
    metadata:
      labels:
        app: {name}
        fix_id: {fix_id}
    spec:
      serviceAccountName: {service_account}
      containers:
      - name: {name}
        image: {image}
        imagePullPolicy: Always
        ports:
        - containerPort: {port}
        resources:
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
          requests:
            cpu: {cpu_request}
            memory: {memory_request}
        livenessProbe:
          httpGet:
            path: {liveness_path}
            port: {liveness_port}
          initialDelaySeconds: {liveness_delay}
          periodSeconds: {liveness_period}
        readinessProbe:
          httpGet:
            path: {readiness_path}
            port: {readiness_port}
          initialDelaySeconds: {readiness_delay}
          periodSeconds: {readiness_period}
        env:
{environment_variables}
""",
            "service.yaml": """
apiVersion: v1
kind: Service
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: {name}
    fix_id: {fix_id}
spec:
  selector:
    app: {name}
    fix_id: {fix_id}
  ports:
  - port: {port}
    targetPort: {target_port}
  type: {service_type}
""",
            "ingress.yaml": """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: {name}
    fix_id: {fix_id}
  annotations:
{annotations}
spec:
  rules:
  - host: {host}
    http:
      paths:
      - path: {path}
        pathType: Prefix
        backend:
          service:
            name: {service_name}
            port:
              number: {service_port}
"""
        }
        
        for name, content in templates.items():
            template_path = self.template_dir / name
            if not template_path.exists():
                with open(template_path, "w") as f:
                    f.write(content)
                    
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
            
    def get_namespace(self) -> str:
        """Get or create the configured Kubernetes namespace.
        
        Returns:
            str: Namespace name
        """
        # Check if namespace exists
        result = self._run_kubectl(["get", "namespace", self.namespace, "-o", "json"])
        
        if not result["success"] and not result.get("simulated", False):
            # Create namespace
            logger.info(f"Creating Kubernetes namespace: {self.namespace}")
            namespace_yaml = f"""
            apiVersion: v1
            kind: Namespace
            metadata:
              name: {self.namespace}
            """
            
            create_result = self._run_kubectl(["apply", "-f", "-"], input_data=namespace_yaml)
            if not create_result["success"] and not create_result.get("simulated", False):
                logger.error(f"Failed to create namespace: {create_result}")
                
        return self.namespace
        
    def apply_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Apply YAML manifest to Kubernetes.
        
        Args:
            yaml_content: YAML manifest
            
        Returns:
            Dict: Result of kubectl apply
        """
        try:
            # Parse YAML to validate and log what we're applying
            resource = yaml.safe_load(yaml_content)
            kind = resource.get("kind", "Unknown")
            name = resource.get("metadata", {}).get("name", "unknown")
            
            logger.info(f"Applying {kind} '{name}' to Kubernetes")
            
            # Apply YAML
            return self._run_kubectl(["apply", "-f", "-"], input_data=yaml_content)
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML: {str(e)}")
            return {"success": False, "error": f"Invalid YAML: {str(e)}"}
            
    def delete_resource(self, resource_type: str, name: str) -> Dict[str, Any]:
        """Delete Kubernetes resource.
        
        Args:
            resource_type: Resource type (deployment, service, etc.)
            name: Resource name
            
        Returns:
            Dict: Result of kubectl delete
        """
        logger.info(f"Deleting {resource_type} '{name}' from namespace {self.namespace}")
        
        return self._run_kubectl([
            "delete", resource_type, name,
            "--namespace", self.namespace
        ])
        
    def get_resource(self, resource_type: str, name: str) -> Dict[str, Any]:
        """Get Kubernetes resource.
        
        Args:
            resource_type: Resource type (deployment, service, etc.)
            name: Resource name
            
        Returns:
            Dict: Resource information
        """
        result = self._run_kubectl([
            "get", resource_type, name,
            "--namespace", self.namespace,
            "-o", "json"
        ])
        
        if result.get("simulated", False):
            # Return simulated resource
            return {
                "success": True,
                "simulated": True,
                "resource": {
                    "kind": resource_type.capitalize(),
                    "metadata": {
                        "name": name,
                        "namespace": self.namespace
                    },
                    "status": {
                        "phase": "Running"
                    }
                }
            }
            
        return result
        
    def wait_for_deployment(self, name: str, timeout: int = 300) -> bool:
        """Wait for deployment to be ready.
        
        Args:
            name: Deployment name
            timeout: Timeout in seconds
            
        Returns:
            bool: True if deployment is ready, False otherwise
        """
        logger.info(f"Waiting for deployment '{name}' to be ready (timeout: {timeout}s)")
        
        if not self.kubectl_available:
            logger.info(f"Simulating wait for deployment '{name}'")
            return True
            
        result = self._run_kubectl([
            "rollout", "status", "deployment", name,
            "--namespace", self.namespace,
            f"--timeout={timeout}s"
        ])
        
        return result["success"]
        
    def create_deployment(self, service_name: str, fix_id: str, 
                         image: str, port: int = 8000, 
                         replicas: int = 1) -> Dict[str, Any]:
        """Create Kubernetes deployment for a service.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            image: Docker image to deploy
            port: Container port
            replicas: Number of replicas
            
        Returns:
            Dict: Result of deployment creation
        """
        # Ensure namespace exists
        self.get_namespace()
        
        # Generate deployment name
        deployment_name = f"{service_name}-{fix_id}"
        
        # Load template
        try:
            with open(self.template_dir / "deployment.yaml", "r") as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(f"Deployment template not found at {self.template_dir / 'deployment.yaml'}")
            return {"success": False, "error": "Deployment template not found"}
            
        # Format environment variables
        env_vars = self.config.get("environment_variables", {})
        env_var_yaml = ""
        for name, value in env_vars.items():
            env_var_yaml += f"        - name: {name}\n          value: \"{value}\"\n"
            
        # Format template
        yaml_content = template.format(
            name=deployment_name,
            namespace=self.namespace,
            fix_id=fix_id,
            replicas=replicas,
            strategy_type=self.deployment_strategy,
            service_account=self.service_account,
            image=image,
            port=port,
            cpu_limit=self.resource_limits.get("cpu", "1.0"),
            memory_limit=self.resource_limits.get("memory", "1g"),
            cpu_request=self.resource_limits.get("cpu_request", "0.5"),
            memory_request=self.resource_limits.get("memory_request", "512m"),
            liveness_path=self.health_probes.get("liveness", {}).get("path", "/health"),
            liveness_port=self.health_probes.get("liveness", {}).get("port", port),
            liveness_delay=self.health_probes.get("liveness", {}).get("initial_delay", 10),
            liveness_period=self.health_probes.get("liveness", {}).get("period", 30),
            readiness_path=self.health_probes.get("readiness", {}).get("path", "/health"),
            readiness_port=self.health_probes.get("readiness", {}).get("port", port),
            readiness_delay=self.health_probes.get("readiness", {}).get("initial_delay", 5),
            readiness_period=self.health_probes.get("readiness", {}).get("period", 10),
            environment_variables=env_var_yaml
        )
        
        # Apply deployment
        result = self.apply_yaml(yaml_content)
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_deployment_created",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_name": deployment_name,
                    "image": image,
                    "namespace": self.namespace,
                    "success": result["success"]
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def create_service(self, service_name: str, fix_id: str, 
                      port: int = 80, target_port: int = 8000,
                      service_type: str = "ClusterIP") -> Dict[str, Any]:
        """Create Kubernetes service for a deployment.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            port: Service port
            target_port: Target container port
            service_type: Service type (ClusterIP, NodePort, LoadBalancer)
            
        Returns:
            Dict: Result of service creation
        """
        # Generate service name
        name = f"{service_name}-{fix_id}"
        
        # Load template
        try:
            with open(self.template_dir / "service.yaml", "r") as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(f"Service template not found at {self.template_dir / 'service.yaml'}")
            return {"success": False, "error": "Service template not found"}
            
        # Format template
        yaml_content = template.format(
            name=name,
            namespace=self.namespace,
            fix_id=fix_id,
            port=port,
            target_port=target_port,
            service_type=service_type
        )
        
        # Apply service
        result = self.apply_yaml(yaml_content)
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_service_created",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "kubernetes_service_name": name,
                    "namespace": self.namespace,
                    "port": port,
                    "target_port": target_port,
                    "service_type": service_type,
                    "success": result["success"]
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def create_ingress(self, service_name: str, fix_id: str, 
                      host: str, path: str = "/",
                      annotations: Dict[str, str] = None) -> Dict[str, Any]:
        """Create Kubernetes ingress for a service.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            host: Ingress host
            path: Ingress path
            annotations: Ingress annotations
            
        Returns:
            Dict: Result of ingress creation
        """
        # Generate names
        ingress_name = f"{service_name}-{fix_id}"
        service_kubernetes_name = f"{service_name}-{fix_id}"
        
        # Set default annotations
        if annotations is None:
            annotations = {}
            
        # Format annotations
        annotations_yaml = ""
        for name, value in annotations.items():
            annotations_yaml += f"    {name}: {value}\n"
            
        # Load template
        try:
            with open(self.template_dir / "ingress.yaml", "r") as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(f"Ingress template not found at {self.template_dir / 'ingress.yaml'}")
            return {"success": False, "error": "Ingress template not found"}
            
        # Format template
        yaml_content = template.format(
            name=ingress_name,
            namespace=self.namespace,
            fix_id=fix_id,
            host=host,
            path=path,
            service_name=service_kubernetes_name,
            service_port=80,
            annotations=annotations_yaml
        )
        
        # Apply ingress
        result = self.apply_yaml(yaml_content)
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_ingress_created",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "ingress_name": ingress_name,
                    "namespace": self.namespace,
                    "host": host,
                    "path": path,
                    "success": result["success"]
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def delete_deployment(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Delete Kubernetes deployment.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            
        Returns:
            Dict: Result of deployment deletion
        """
        deployment_name = f"{service_name}-{fix_id}"
        result = self.delete_resource("deployment", deployment_name)
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_deployment_deleted",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_name": deployment_name,
                    "namespace": self.namespace,
                    "success": result["success"]
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def delete_service(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Delete Kubernetes service.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            
        Returns:
            Dict: Result of service deletion
        """
        service_name = f"{service_name}-{fix_id}"
        result = self.delete_resource("service", service_name)
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_service_deleted",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "kubernetes_service_name": service_name,
                    "namespace": self.namespace,
                    "success": result["success"]
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def delete_ingress(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Delete Kubernetes ingress.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            
        Returns:
            Dict: Result of ingress deletion
        """
        ingress_name = f"{service_name}-{fix_id}"
        result = self.delete_resource("ingress", ingress_name)
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_ingress_deleted",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "ingress_name": ingress_name,
                    "namespace": self.namespace,
                    "success": result["success"]
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return result
        
    def deploy_service(self, service_name: str, fix_id: str, image: str,
                     host: str = None, create_ingress: bool = True) -> Dict[str, Any]:
        """Deploy a service to Kubernetes.
        
        This is a convenience function that creates a deployment, service, and
        optionally an ingress for a service.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            image: Docker image to deploy
            host: Ingress host (if None, ingress will not be created)
            create_ingress: Whether to create ingress
            
        Returns:
            Dict: Results of deployment operations
        """
        results = {}
        
        # Create deployment
        deployment_result = self.create_deployment(
            service_name=service_name,
            fix_id=fix_id,
            image=image
        )
        results["deployment"] = deployment_result
        
        if not deployment_result["success"] and not deployment_result.get("simulated", False):
            logger.error(f"Failed to create deployment for {service_name}/{fix_id}")
            return results
            
        # Create service
        service_result = self.create_service(
            service_name=service_name,
            fix_id=fix_id
        )
        results["service"] = service_result
        
        if not service_result["success"] and not service_result.get("simulated", False):
            logger.error(f"Failed to create service for {service_name}/{fix_id}")
            return results
            
        # Create ingress if host is provided
        if create_ingress and host:
            ingress_result = self.create_ingress(
                service_name=service_name,
                fix_id=fix_id,
                host=host
            )
            results["ingress"] = ingress_result
            
            if not ingress_result["success"] and not ingress_result.get("simulated", False):
                logger.error(f"Failed to create ingress for {service_name}/{fix_id}")
                
        # Wait for deployment to be ready
        wait_result = self.wait_for_deployment(f"{service_name}-{fix_id}")
        results["wait"] = {"success": wait_result}
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_service_deployed",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "image": image,
                    "namespace": self.namespace,
                    "success": deployment_result["success"] and service_result["success"],
                    "host": host if host else "none"
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return results
        
    def undeploy_service(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Undeploy a service from Kubernetes.
        
        This is a convenience function that deletes a deployment, service, and
        ingress for a service.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            
        Returns:
            Dict: Results of undeployment operations
        """
        results = {}
        
        # Delete ingress
        ingress_result = self.delete_ingress(service_name, fix_id)
        results["ingress"] = ingress_result
        
        # Delete service
        service_result = self.delete_service(service_name, fix_id)
        results["service"] = service_result
        
        # Delete deployment
        deployment_result = self.delete_deployment(service_name, fix_id)
        results["deployment"] = deployment_result
        
        # Log to audit
        try:
            get_audit_logger().log_event(
                event_type="kubernetes_service_undeployed",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "namespace": self.namespace,
                    "success": deployment_result["success"] and service_result["success"]
                }
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")
            
        return results


# Singleton instance
_kubernetes_deployment = None

def get_kubernetes_deployment(config: Dict[str, Any] = None) -> KubernetesDeployment:
    """Get or create the singleton KubernetesDeployment instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        KubernetesDeployment: Singleton instance
    """
    global _kubernetes_deployment
    if _kubernetes_deployment is None:
        _kubernetes_deployment = KubernetesDeployment(config)
    return _kubernetes_deployment