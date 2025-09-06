"""
Istio Service Mesh integration for Homeostasis.

Provides functionality for integration with Istio service mesh, enabling
advanced traffic management, observability, and security features for
self-healing deployments.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class IstioIntegration:
    """
    Integration with Istio service mesh for advanced traffic management.

    Provides functionality to create and manage Istio resources for traffic management,
    including VirtualServices, DestinationRules, and ServiceEntries.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Istio integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Set default values from config
        self.namespace = self.config.get("namespace", "default")
        self.istio_version = self.config.get("istio_version", "1.18.0")
        self.template_dir = Path(
            self.config.get("template_dir", "modules/deployment/service_mesh/templates")
        )

        # Ensure template directory exists
        os.makedirs(self.template_dir, exist_ok=True)

        # Check if istioctl is available
        self.istioctl_available = self._check_istioctl_available()
        if not self.istioctl_available:
            logger.warning("istioctl not found, Istio operations will be simulated")

        # Create templates if they don't exist
        self._ensure_templates_exist()

    def _check_istioctl_available(self) -> bool:
        """Check if istioctl is available.

        Returns:
            bool: True if istioctl is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "istioctl"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _ensure_templates_exist(self) -> None:
        """Create default templates if they don't exist."""
        templates = {
            "virtual_service.yaml": """
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: {name}
  namespace: {namespace}
spec:
  hosts:
  - {host}
  gateways:
  - {gateway}
  http:
  - route:
    - destination:
        host: {service_name}
        subset: {subset}
      weight: {weight}
    - destination:
        host: {service_name}
        subset: {canary_subset}
      weight: {canary_weight}
""",
            "destination_rule.yaml": """
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: {name}
  namespace: {namespace}
spec:
  host: {service_name}
  subsets:
  - name: {subset}
    labels:
      version: {version}
  - name: {canary_subset}
    labels:
      version: {canary_version}
""",
            "service_entry.yaml": """
apiVersion: networking.istio.io/v1alpha3
kind: ServiceEntry
metadata:
  name: {name}
  namespace: {namespace}
spec:
  hosts:
  - {host}
  ports:
  - number: {port}
    name: {protocol}
    protocol: {protocol}
  resolution: {resolution}
  location: {location}
""",
        }

        for name, content in templates.items():
            template_path = self.template_dir / name
            if not template_path.exists():
                with open(template_path, "w") as f:
                    f.write(content)

    def _run_istioctl(
        self, args: List[str], input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run istioctl command.

        Args:
            args: List of arguments for istioctl
            input_data: Optional input data for istioctl

        Returns:
            Dict: Result of istioctl command
        """
        if not self.istioctl_available:
            logger.info(f"Simulating istioctl command: istioctl {' '.join(args)}")
            return {"success": True, "simulated": True}

        try:
            cmd = ["istioctl"] + args
            logger.debug(f"Running istioctl command: {' '.join(cmd)}")

            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""

            if process.returncode != 0:
                logger.error(f"istioctl command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
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
            logger.exception(f"Error running istioctl command: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_kubectl(
        self, args: List[str], input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run kubectl command.

        Args:
            args: List of arguments for kubectl
            input_data: Optional input data for kubectl

        Returns:
            Dict: Result of kubectl command
        """
        try:
            cmd = ["kubectl"] + args
            logger.debug(f"Running kubectl command: {' '.join(cmd)}")

            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""

            if process.returncode != 0:
                logger.error(f"kubectl command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
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

            logger.info(f"Applying Istio {kind} '{name}' to Kubernetes")

            # Apply YAML
            return self._run_kubectl(["apply", "-f", "-"], input_data=yaml_content)

        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML: {str(e)}")
            return {"success": False, "error": f"Invalid YAML: {str(e)}"}

    def create_virtual_service(
        self,
        service_name: str,
        fix_id: str,
        host: str,
        gateway: str = "mesh",
        canary_percentage: int = 0,
    ) -> Dict[str, Any]:
        """Create an Istio VirtualService for canary routing.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            host: VirtualService host
            gateway: Gateway name
            canary_percentage: Percentage of traffic to route to canary (0-100)

        Returns:
            Dict: Result of operation
        """
        # Generate names
        vs_name = f"{service_name}-vs"

        # Calculate weights
        canary_weight = canary_percentage
        primary_weight = 100 - canary_weight

        # Generate subset names
        primary_subset = "primary"
        canary_subset = f"canary-{fix_id}"

        # Load template
        try:
            with open(self.template_dir / "virtual_service.yaml", "r") as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(
                f"Virtual service template not found at {self.template_dir / 'virtual_service.yaml'}"
            )
            return {"success": False, "error": "Template not found"}

        # Format template
        yaml_content = template.format(
            name=vs_name,
            namespace=self.namespace,
            host=host,
            gateway=gateway,
            service_name=service_name,
            subset=primary_subset,
            weight=primary_weight,
            canary_subset=canary_subset,
            canary_weight=canary_weight,
        )

        # Apply the VirtualService
        result = self.apply_yaml(yaml_content)

        # Log the operation
        try:
            get_audit_logger().log_event(
                event_type="istio_virtual_service_created",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "virtual_service_name": vs_name,
                    "primary_weight": primary_weight,
                    "canary_weight": canary_weight,
                    "success": result["success"],
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        return result

    def create_destination_rule(
        self,
        service_name: str,
        fix_id: str,
        primary_version: str = "v1",
        canary_version: str = "canary",
    ) -> Dict[str, Any]:
        """Create an Istio DestinationRule for service subsets.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            primary_version: Version label for primary subset
            canary_version: Version label for canary subset

        Returns:
            Dict: Result of operation
        """
        # Generate names
        dr_name = f"{service_name}-dr"

        # Generate subset names
        primary_subset = "primary"
        canary_subset = f"canary-{fix_id}"

        # Load template
        try:
            with open(self.template_dir / "destination_rule.yaml", "r") as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(
                f"Destination rule template not found at {self.template_dir / 'destination_rule.yaml'}"
            )
            return {"success": False, "error": "Template not found"}

        # Format template
        yaml_content = template.format(
            name=dr_name,
            namespace=self.namespace,
            service_name=service_name,
            subset=primary_subset,
            version=primary_version,
            canary_subset=canary_subset,
            canary_version=canary_version,
        )

        # Apply the DestinationRule
        result = self.apply_yaml(yaml_content)

        # Log the operation
        try:
            get_audit_logger().log_event(
                event_type="istio_destination_rule_created",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "destination_rule_name": dr_name,
                    "primary_subset": primary_subset,
                    "canary_subset": canary_subset,
                    "success": result["success"],
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        return result

    def update_canary_percentage(
        self, service_name: str, percentage: int
    ) -> Dict[str, Any]:
        """Update canary traffic percentage for a service.

        Args:
            service_name: Name of the service
            percentage: Percentage of traffic to route to canary (0-100)

        Returns:
            Dict: Result of operation
        """
        # Generate VirtualService name
        vs_name = f"{service_name}-vs"

        # Get current VirtualService
        result = self._run_kubectl(
            [
                "get",
                "virtualservice",
                vs_name,
                "--namespace",
                self.namespace,
                "-o",
                "yaml",
            ]
        )

        if not result["success"]:
            logger.error(f"Failed to get VirtualService {vs_name}")
            return result

        try:
            # Parse YAML
            if result.get("simulated", False):
                # Simulated result, just assume success
                logger.info(
                    f"Simulating update of canary percentage to {percentage}% for {service_name}"
                )
                return {"success": True, "simulated": True}

            vs_yaml = yaml.safe_load(result["output"])

            # Update weights
            canary_weight = percentage
            primary_weight = 100 - canary_weight

            http_routes = vs_yaml.get("spec", {}).get("http", [{}])[0]
            routes = http_routes.get("route", [])

            if len(routes) >= 2:
                routes[0]["weight"] = primary_weight
                routes[1]["weight"] = canary_weight

                # Convert back to YAML
                updated_yaml = yaml.dump(vs_yaml)

                # Apply the updated VirtualService
                apply_result = self.apply_yaml(updated_yaml)

                # Log the operation
                try:
                    get_audit_logger().log_event(
                        event_type="istio_canary_percentage_updated",
                        details={
                            "service_name": service_name,
                            "virtual_service_name": vs_name,
                            "primary_weight": primary_weight,
                            "canary_weight": canary_weight,
                            "success": apply_result["success"],
                        },
                    )
                except Exception as e:
                    logger.debug(f"Could not log to audit log: {str(e)}")

                return apply_result
            else:
                logger.error(
                    f"VirtualService {vs_name} does not have expected route structure"
                )
                return {"success": False, "error": "Invalid VirtualService structure"}

        except Exception as e:
            logger.error(f"Error updating canary percentage: {str(e)}")
            return {"success": False, "error": str(e)}

    def delete_istio_resources(self, service_name: str) -> Dict[str, Any]:
        """Delete Istio resources for a service.

        Args:
            service_name: Name of the service

        Returns:
            Dict: Result of operation
        """
        # Generate resource names
        vs_name = f"{service_name}-vs"
        dr_name = f"{service_name}-dr"

        # Delete VirtualService
        vs_result = self._run_kubectl(
            ["delete", "virtualservice", vs_name, "--namespace", self.namespace]
        )

        # Delete DestinationRule
        dr_result = self._run_kubectl(
            ["delete", "destinationrule", dr_name, "--namespace", self.namespace]
        )

        # Log the operation
        try:
            get_audit_logger().log_event(
                event_type="istio_resources_deleted",
                details={
                    "service_name": service_name,
                    "virtual_service_name": vs_name,
                    "destination_rule_name": dr_name,
                    "success": vs_result["success"] and dr_result["success"],
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        return {"virtualservice": vs_result, "destinationrule": dr_result}

    def setup_canary_deployment(
        self, service_name: str, fix_id: str, host: str, initial_percentage: int = 0
    ) -> Dict[str, Any]:
        """Setup Istio resources for canary deployment.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            host: VirtualService host
            initial_percentage: Initial percentage of traffic to canary

        Returns:
            Dict: Result of operation
        """
        # Create DestinationRule for subsets
        dr_result = self.create_destination_rule(service_name, fix_id)

        if not dr_result["success"] and not dr_result.get("simulated", False):
            logger.error(f"Failed to create DestinationRule for {service_name}")
            return dr_result

        # Create VirtualService for traffic routing
        vs_result = self.create_virtual_service(
            service_name=service_name,
            fix_id=fix_id,
            host=host,
            canary_percentage=initial_percentage,
        )

        if not vs_result["success"] and not vs_result.get("simulated", False):
            logger.error(f"Failed to create VirtualService for {service_name}")
            return vs_result

        logger.info(
            f"Successfully setup Istio canary deployment for {service_name} with {initial_percentage}% traffic to fix {fix_id}"
        )

        return {
            "success": True,
            "destinationrule": dr_result,
            "virtualservice": vs_result,
        }

    def complete_canary_deployment(self, service_name: str) -> Dict[str, Any]:
        """Complete canary deployment by setting 100% traffic to canary.

        Args:
            service_name: Name of the service

        Returns:
            Dict: Result of operation
        """
        return self.update_canary_percentage(service_name, 100)

    def rollback_canary_deployment(self, service_name: str) -> Dict[str, Any]:
        """Rollback canary deployment by setting 0% traffic to canary.

        Args:
            service_name: Name of the service

        Returns:
            Dict: Result of operation
        """
        return self.update_canary_percentage(service_name, 0)


# Singleton instance
_istio_integration = None


def get_istio_integration(config: Dict[str, Any] = None) -> IstioIntegration:
    """Get or create the singleton IstioIntegration instance.

    Args:
        config: Optional configuration

    Returns:
        IstioIntegration: Singleton instance
    """
    global _istio_integration
    if _istio_integration is None:
        _istio_integration = IstioIntegration(config)
    return _istio_integration
