"""
Azure Provider for Homeostasis.

This module provides integration with Microsoft Azure for deploying fixes.
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

from modules.deployment.cloud.base_provider import BaseCloudProvider
from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class AzureProvider(BaseCloudProvider):
    """
    Microsoft Azure provider implementation.

    Supports deploying to:
    - Function Apps
    - Container Instances
    - AKS (Azure Kubernetes Service)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure provider.

        Args:
            config: Azure configuration dictionary
        """
        super().__init__(config)

        # Azure-specific configuration
        self.subscription_id = self.config.get("subscription_id")
        self.resource_group = self.config.get("resource_group")
        self.location = self.region or self.config.get("location", "eastus")
        self.function_app_enabled = self.config.get("function_app", False)
        self.container_instance_enabled = self.config.get("container_instance", False)
        self.aks_enabled = self.config.get("aks_cluster", False)

        # Default deployment type if multiple are enabled
        self.default_deployment_type = None
        if self.function_app_enabled:
            self.default_deployment_type = "function"
        elif self.container_instance_enabled:
            self.default_deployment_type = "container"
        elif self.aks_enabled:
            self.default_deployment_type = "aks"

        # Check if Azure CLI is available
        self.az_available = self._check_az_available()
        if not self.az_available:
            logger.warning("Azure CLI not found, Azure operations will be simulated")

    def _check_az_available(self) -> bool:
        """Check if Azure CLI is available.

        Returns:
            bool: True if Azure CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "az"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_az(
        self, args: List[str], input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run Azure CLI command.

        Args:
            args: Azure CLI arguments
            input_data: Optional input data

        Returns:
            Dict[str, Any]: Command result
        """
        if not self.az_available:
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
                check=False,
            )

            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""

            if process.returncode != 0:
                logger.error(f"Azure CLI command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
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

    def is_available(self) -> bool:
        """Check if Azure provider is available.

        Returns:
            bool: True if Azure provider is available, False otherwise
        """
        if not self.az_available:
            return False

        if not self.subscription_id:
            logger.error("Azure subscription ID not configured")
            return False

        # Try to get subscription info
        result = self._run_az(["account", "show"])
        return result.get("success", False)

    def _deploy_function(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to Azure Function App.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        app_name = kwargs.get("app_name", f"{service_name}-{fix_id}")
        plan_name = kwargs.get("plan_name", f"{app_name}-plan")
        storage_name = kwargs.get("storage_name", f"{app_name}storage".replace("-", ""))
        runtime = kwargs.get("runtime", "python")

        if not self.az_available:
            logger.info(f"Simulating Azure Function App deployment: {app_name}")
            return {
                "success": True,
                "simulated": True,
                "app_name": app_name,
                "deployment_type": "function",
            }

        # Check if source path exists
        if not os.path.exists(source_path):
            logger.error(f"Source path not found: {source_path}")
            return {"success": False, "error": f"Source path not found: {source_path}"}

        try:
            # Check if resource group exists
            if not self.resource_group:
                logger.error("Azure resource group not configured")
                return {
                    "success": False,
                    "error": "Azure resource group not configured",
                }

            # Check if resource group exists
            check_rg_result = self._run_az(
                ["group", "exists", "--name", self.resource_group]
            )

            if not check_rg_result.get("success", False):
                return check_rg_result

            rg_exists = check_rg_result in [
                True,
                "true",
                "True",
            ] or check_rg_result.get("output") in [True, "true", "True"]

            if not rg_exists:
                # Create resource group
                logger.info(f"Creating Azure resource group: {self.resource_group}")
                create_rg_result = self._run_az(
                    [
                        "group",
                        "create",
                        "--name",
                        self.resource_group,
                        "--location",
                        self.location,
                    ]
                )

                if not create_rg_result.get("success", False):
                    return create_rg_result

            # Check if storage account exists
            check_storage_result = self._run_az(
                ["storage", "account", "check-name", "--name", storage_name]
            )

            if check_storage_result.get(
                "nameAvailable", True
            ) or check_storage_result.get("simulated", False):
                # Create storage account
                logger.info(f"Creating Azure storage account: {storage_name}")
                create_storage_result = self._run_az(
                    [
                        "storage",
                        "account",
                        "create",
                        "--name",
                        storage_name,
                        "--resource-group",
                        self.resource_group,
                        "--location",
                        self.location,
                        "--sku",
                        "Standard_LRS",
                    ]
                )

                if not create_storage_result.get(
                    "success", False
                ) and not create_storage_result.get("simulated", False):
                    return create_storage_result

            # Check if function app exists
            check_function_result = self._run_az(
                [
                    "functionapp",
                    "show",
                    "--name",
                    app_name,
                    "--resource-group",
                    self.resource_group,
                ]
            )

            if check_function_result.get(
                "success", False
            ) and not check_function_result.get("simulated", False):
                # Function app exists, deploy code
                logger.info(f"Deploying code to existing Function App: {app_name}")

                deploy_result = self._run_az(
                    [
                        "functionapp",
                        "deployment",
                        "source",
                        "config-zip",
                        "--name",
                        app_name,
                        "--resource-group",
                        self.resource_group,
                        "--src",
                        source_path,
                    ]
                )

                return deploy_result

            else:
                # Create consumption plan
                logger.info(f"Creating Azure Function App plan: {plan_name}")
                create_plan_result = self._run_az(
                    [
                        "functionapp",
                        "plan",
                        "create",
                        "--name",
                        plan_name,
                        "--resource-group",
                        self.resource_group,
                        "--location",
                        self.location,
                        "--sku",
                        "Y1",
                        "--number-of-workers",
                        "1",
                    ]
                )

                if not create_plan_result.get(
                    "success", False
                ) and not create_plan_result.get("simulated", False):
                    return create_plan_result

                # Create function app
                logger.info(f"Creating Azure Function App: {app_name}")
                create_function_result = self._run_az(
                    [
                        "functionapp",
                        "create",
                        "--name",
                        app_name,
                        "--resource-group",
                        self.resource_group,
                        "--plan",
                        plan_name,
                        "--storage-account",
                        storage_name,
                        "--runtime",
                        runtime,
                    ]
                )

                if not create_function_result.get(
                    "success", False
                ) and not create_function_result.get("simulated", False):
                    return create_function_result

                # Deploy code
                logger.info(f"Deploying code to new Function App: {app_name}")
                deploy_result = self._run_az(
                    [
                        "functionapp",
                        "deployment",
                        "source",
                        "config-zip",
                        "--name",
                        app_name,
                        "--resource-group",
                        self.resource_group,
                        "--src",
                        source_path,
                    ]
                )

                return deploy_result

        except Exception as e:
            logger.exception(f"Error deploying to Azure Function App: {str(e)}")
            return {"success": False, "error": str(e)}

    def _deploy_container(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to Azure Container Instances.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        container_group_name = kwargs.get(
            "container_group_name", f"{service_name}-{fix_id}"
        )
        image = kwargs.get("image")

        if not image:
            logger.error("Image required for Container Instance deployment")
            return {
                "success": False,
                "error": "Image required for Container Instance deployment",
            }

        if not self.az_available:
            logger.info(
                f"Simulating Azure Container Instance deployment: {container_group_name}"
            )
            return {
                "success": True,
                "simulated": True,
                "container_group_name": container_group_name,
                "deployment_type": "container",
            }

        try:
            # Check if resource group exists
            if not self.resource_group:
                logger.error("Azure resource group not configured")
                return {
                    "success": False,
                    "error": "Azure resource group not configured",
                }

            # Check if resource group exists
            check_rg_result = self._run_az(
                ["group", "exists", "--name", self.resource_group]
            )

            if not check_rg_result.get("success", False):
                return check_rg_result

            rg_exists = check_rg_result in [
                True,
                "true",
                "True",
            ] or check_rg_result.get("output") in [True, "true", "True"]

            if not rg_exists:
                # Create resource group
                logger.info(f"Creating Azure resource group: {self.resource_group}")
                create_rg_result = self._run_az(
                    [
                        "group",
                        "create",
                        "--name",
                        self.resource_group,
                        "--location",
                        self.location,
                    ]
                )

                if not create_rg_result.get("success", False):
                    return create_rg_result

            # Check if container group exists
            check_container_result = self._run_az(
                [
                    "container",
                    "show",
                    "--name",
                    container_group_name,
                    "--resource-group",
                    self.resource_group,
                ]
            )

            if check_container_result.get(
                "success", False
            ) and not check_container_result.get("simulated", False):
                # Container group exists, delete it first
                logger.info(
                    f"Deleting existing Container Instance: {container_group_name}"
                )
                delete_result = self._run_az(
                    [
                        "container",
                        "delete",
                        "--name",
                        container_group_name,
                        "--resource-group",
                        self.resource_group,
                        "--yes",
                    ]
                )

                if not delete_result.get("success", False) and not delete_result.get(
                    "simulated", False
                ):
                    return delete_result

            # Create container group
            logger.info(f"Creating Azure Container Instance: {container_group_name}")

            # Set container parameters
            cpu = kwargs.get("cpu", "1.0")
            memory = kwargs.get("memory", "1.5")
            dns_name = kwargs.get("dns_name", container_group_name)
            ports = kwargs.get("ports", [80])
            env_vars = kwargs.get("env_vars", {})

            # Build command
            cmd = [
                "container",
                "create",
                "--name",
                container_group_name,
                "--resource-group",
                self.resource_group,
                "--image",
                image,
                "--cpu",
                cpu,
                "--memory",
                memory,
                "--location",
                self.location,
            ]

            # Add ports
            for port in ports:
                cmd.extend(["--port", str(port)])

            # Add DNS name if provided
            if dns_name:
                cmd.extend(["--dns-name-label", dns_name])

            # Add environment variables
            for name, value in env_vars.items():
                cmd.extend(["--environment-variables", f"{name}={value}"])

            # Create container group
            create_result = self._run_az(cmd)

            return create_result

        except Exception as e:
            logger.exception(f"Error deploying to Azure Container Instance: {str(e)}")
            return {"success": False, "error": str(e)}

    def _deploy_aks(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to Azure Kubernetes Service.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        cluster_name = kwargs.get("cluster_name", "homeostasis")
        service_identifier = f"{service_name}-{fix_id}"
        image = kwargs.get("image")

        if not image:
            logger.error("Image required for AKS deployment")
            return {"success": False, "error": "Image required for AKS deployment"}

        if not self.az_available:
            logger.info(f"Simulating Azure AKS deployment: {service_identifier}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_identifier,
                "cluster": cluster_name,
                "deployment_type": "aks",
            }

        try:
            # Check if resource group exists
            if not self.resource_group:
                logger.error("Azure resource group not configured")
                return {
                    "success": False,
                    "error": "Azure resource group not configured",
                }

            # Get AKS credentials
            logger.info(f"Getting AKS credentials for cluster: {cluster_name}")
            get_credentials_result = self._run_az(
                [
                    "aks",
                    "get-credentials",
                    "--resource-group",
                    self.resource_group,
                    "--name",
                    cluster_name,
                    "--overwrite-existing",
                ]
            )

            if not get_credentials_result.get(
                "success", False
            ) and not get_credentials_result.get("simulated", False):
                return get_credentials_result

            # Now we can use kubectl
            from modules.deployment.kubernetes.kubernetes_deployment import (
                KubernetesDeployment,
            )

            # Create Kubernetes deployment
            k8s_deployment = KubernetesDeployment(
                {"namespace": kwargs.get("namespace", "default")}
            )

            # Deploy to Kubernetes
            deploy_result = k8s_deployment.deploy_service(
                service_name=service_name,
                fix_id=fix_id,
                image=image,
                host=kwargs.get("host"),
            )

            return {
                "success": True,
                "service_name": service_identifier,
                "cluster": cluster_name,
                "deployment_type": "aks",
                "kubernetes_result": deploy_result,
            }

        except Exception as e:
            logger.exception(f"Error deploying to Azure AKS: {str(e)}")
            return {"success": False, "error": str(e)}

    def deploy_service(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy a service to Azure.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional provider-specific parameters
                - deployment_type: "function", "container", or "aks"
                - For Function Apps: app_name, plan_name, storage_name, runtime
                - For Container Instances: container_group_name, image, cpu, memory, dns_name, ports, env_vars
                - For AKS: cluster_name, image, namespace, host

        Returns:
            Dict[str, Any]: Deployment information
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        result = None

        if deployment_type == "function" and self.function_app_enabled:
            result = self._deploy_function(service_name, fix_id, source_path, **kwargs)
        elif deployment_type == "container" and self.container_instance_enabled:
            result = self._deploy_container(service_name, fix_id, source_path, **kwargs)
        elif deployment_type == "aks" and self.aks_enabled:
            result = self._deploy_aks(service_name, fix_id, source_path, **kwargs)
        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return {
                "success": False,
                "error": f"Unsupported or disabled deployment type: {deployment_type}",
            }

        # Log deployment to audit
        try:
            get_audit_logger().log_event(
                event_type="azure_deployment",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                    "subscription_id": self.subscription_id,
                    "resource_group": self.resource_group,
                    "location": self.location,
                    "success": result.get("success", False),
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        return result

    def _undeploy_function(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy from Azure Function App.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        app_name = kwargs.get("app_name", f"{service_name}-{fix_id}")

        if not self.az_available:
            logger.info(f"Simulating Azure Function App undeployment: {app_name}")
            return {"success": True, "simulated": True, "app_name": app_name}

        # Check if resource group exists
        if not self.resource_group:
            logger.error("Azure resource group not configured")
            return {"success": False, "error": "Azure resource group not configured"}

        # Delete function app
        logger.info(f"Deleting Azure Function App: {app_name}")
        return self._run_az(
            [
                "functionapp",
                "delete",
                "--name",
                app_name,
                "--resource-group",
                self.resource_group,
            ]
        )

    def _undeploy_container(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy from Azure Container Instances.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        container_group_name = kwargs.get(
            "container_group_name", f"{service_name}-{fix_id}"
        )

        if not self.az_available:
            logger.info(
                f"Simulating Azure Container Instance undeployment: {container_group_name}"
            )
            return {
                "success": True,
                "simulated": True,
                "container_group_name": container_group_name,
            }

        # Check if resource group exists
        if not self.resource_group:
            logger.error("Azure resource group not configured")
            return {"success": False, "error": "Azure resource group not configured"}

        # Delete container group
        logger.info(f"Deleting Azure Container Instance: {container_group_name}")
        return self._run_az(
            [
                "container",
                "delete",
                "--name",
                container_group_name,
                "--resource-group",
                self.resource_group,
                "--yes",
            ]
        )

    def _undeploy_aks(self, service_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Undeploy from Azure Kubernetes Service.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        cluster_name = kwargs.get("cluster_name", "homeostasis")
        namespace = kwargs.get("namespace", "default")

        if not self.az_available:
            logger.info(f"Simulating Azure AKS undeployment: {service_name}-{fix_id}")
            return {
                "success": True,
                "simulated": True,
                "service_name": f"{service_name}-{fix_id}",
                "cluster": cluster_name,
                "namespace": namespace,
            }

        try:
            # Check if resource group exists
            if not self.resource_group:
                logger.error("Azure resource group not configured")
                return {
                    "success": False,
                    "error": "Azure resource group not configured",
                }

            # Get AKS credentials
            logger.info(f"Getting AKS credentials for cluster: {cluster_name}")
            get_credentials_result = self._run_az(
                [
                    "aks",
                    "get-credentials",
                    "--resource-group",
                    self.resource_group,
                    "--name",
                    cluster_name,
                    "--overwrite-existing",
                ]
            )

            if not get_credentials_result.get(
                "success", False
            ) and not get_credentials_result.get("simulated", False):
                return get_credentials_result

            # Now we can use kubectl
            from modules.deployment.kubernetes.kubernetes_deployment import (
                KubernetesDeployment,
            )

            # Create Kubernetes deployment
            k8s_deployment = KubernetesDeployment({"namespace": namespace})

            # Undeploy from Kubernetes
            undeploy_result = k8s_deployment.undeploy_service(
                service_name=service_name, fix_id=fix_id
            )

            return {
                "success": True,
                "service_name": f"{service_name}-{fix_id}",
                "cluster": cluster_name,
                "namespace": namespace,
                "kubernetes_result": undeploy_result,
            }

        except Exception as e:
            logger.exception(f"Error undeploying from Azure AKS: {str(e)}")
            return {"success": False, "error": str(e)}

    def undeploy_service(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy a service from Azure.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters
                - deployment_type: "function", "container", or "aks"
                - For Function Apps: app_name
                - For Container Instances: container_group_name
                - For AKS: cluster_name, namespace

        Returns:
            Dict[str, Any]: Undeployment information
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        result = None

        if deployment_type == "function" and self.function_app_enabled:
            result = self._undeploy_function(service_name, fix_id, **kwargs)
        elif deployment_type == "container" and self.container_instance_enabled:
            result = self._undeploy_container(service_name, fix_id, **kwargs)
        elif deployment_type == "aks" and self.aks_enabled:
            result = self._undeploy_aks(service_name, fix_id, **kwargs)
        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return {
                "success": False,
                "error": f"Unsupported or disabled deployment type: {deployment_type}",
            }

        # Log undeployment to audit
        try:
            get_audit_logger().log_event(
                event_type="azure_undeployment",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                    "subscription_id": self.subscription_id,
                    "resource_group": self.resource_group,
                    "success": result.get("success", False),
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        return result

    def get_service_status(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Get status of a deployed service.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters
                - deployment_type: "function", "container", or "aks"

        Returns:
            Dict[str, Any]: Service status information
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        if not self.az_available:
            logger.info(f"Simulating Azure status check for {service_name}-{fix_id}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_name,
                "fix_id": fix_id,
                "deployment_type": deployment_type,
                "status": "Running",
            }

        # Check if resource group exists
        if not self.resource_group:
            logger.error("Azure resource group not configured")
            return {"success": False, "error": "Azure resource group not configured"}

        if deployment_type == "function" and self.function_app_enabled:
            app_name = kwargs.get("app_name", f"{service_name}-{fix_id}")
            logger.info(f"Getting Azure Function App status: {app_name}")

            result = self._run_az(
                [
                    "functionapp",
                    "show",
                    "--name",
                    app_name,
                    "--resource-group",
                    self.resource_group,
                ]
            )

            return result

        elif deployment_type == "container" and self.container_instance_enabled:
            container_group_name = kwargs.get(
                "container_group_name", f"{service_name}-{fix_id}"
            )

            logger.info(
                f"Getting Azure Container Instance status: {container_group_name}"
            )

            result = self._run_az(
                [
                    "container",
                    "show",
                    "--name",
                    container_group_name,
                    "--resource-group",
                    self.resource_group,
                ]
            )

            return result

        elif deployment_type == "aks" and self.aks_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            namespace = kwargs.get("namespace", "default")

            logger.info(f"Getting Azure AKS deployment status: {service_name}-{fix_id}")

            try:
                # Get AKS credentials
                get_credentials_result = self._run_az(
                    [
                        "aks",
                        "get-credentials",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        cluster_name,
                        "--overwrite-existing",
                    ]
                )

                if not get_credentials_result.get(
                    "success", False
                ) and not get_credentials_result.get("simulated", False):
                    return get_credentials_result

                # Now we can use kubectl
                from modules.deployment.kubernetes.kubernetes_deployment import (
                    KubernetesDeployment,
                )

                # Create Kubernetes deployment
                k8s_deployment = KubernetesDeployment({"namespace": namespace})

                # Get Kubernetes resource status
                deployment_status = k8s_deployment.get_resource(
                    resource_type="deployment", name=f"{service_name}-{fix_id}"
                )

                service_status = k8s_deployment.get_resource(
                    resource_type="service", name=f"{service_name}-{fix_id}"
                )

                return {
                    "success": True,
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": "aks",
                    "cluster": cluster_name,
                    "namespace": namespace,
                    "deployment": deployment_status,
                    "service": service_status,
                }

            except Exception as e:
                logger.exception(f"Error getting Azure AKS deployment status: {str(e)}")
                return {"success": False, "error": str(e)}

        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return {
                "success": False,
                "error": f"Unsupported or disabled deployment type: {deployment_type}",
            }

    def get_service_logs(
        self, service_name: str, fix_id: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get logs for a deployed service.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters
                - deployment_type: "function", "container", or "aks"
                - For Function Apps: app_name
                - For Container Instances: container_group_name
                - For AKS: cluster_name, namespace, container, tail

        Returns:
            List[Dict[str, Any]]: Service logs
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        if not self.az_available:
            logger.info(f"Simulating Azure logs retrieval for {service_name}-{fix_id}")
            return [
                {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "message": "Simulated log message",
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                }
            ]

        # Check if resource group exists
        if not self.resource_group:
            logger.error("Azure resource group not configured")
            return []

        if deployment_type == "function" and self.function_app_enabled:
            app_name = kwargs.get("app_name", f"{service_name}-{fix_id}")
            logger.info(f"Getting Azure Function App logs: {app_name}")

            result = self._run_az(
                [
                    "functionapp",
                    "log",
                    "tail",
                    "--name",
                    app_name,
                    "--resource-group",
                    self.resource_group,
                ]
            )

            if not result.get("success", False):
                return []

            logs = result.get("output", "").splitlines()

            # Format logs
            formatted_logs = []
            for i, log in enumerate(logs):
                formatted_logs.append(
                    {
                        "timestamp": "2023-01-01T00:00:00Z",  # Placeholder timestamp
                        "message": log,
                        "service_name": service_name,
                        "fix_id": fix_id,
                        "deployment_type": "function",
                        "index": i,
                    }
                )

            return formatted_logs

        elif deployment_type == "container" and self.container_instance_enabled:
            container_group_name = kwargs.get(
                "container_group_name", f"{service_name}-{fix_id}"
            )
            container_name = kwargs.get("container_name", container_group_name)

            logger.info(
                f"Getting Azure Container Instance logs: {container_group_name}"
            )

            result = self._run_az(
                [
                    "container",
                    "logs",
                    "--name",
                    container_group_name,
                    "--resource-group",
                    self.resource_group,
                    "--container-name",
                    container_name,
                ]
            )

            if not result.get("success", False):
                return []

            logs = result.get("output", "").splitlines()

            # Format logs
            formatted_logs = []
            for i, log in enumerate(logs):
                formatted_logs.append(
                    {
                        "timestamp": "2023-01-01T00:00:00Z",  # Placeholder timestamp
                        "message": log,
                        "service_name": service_name,
                        "fix_id": fix_id,
                        "deployment_type": "container",
                        "index": i,
                    }
                )

            return formatted_logs

        elif deployment_type == "aks" and self.aks_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            namespace = kwargs.get("namespace", "default")

            logger.info(f"Getting Azure AKS deployment logs: {service_name}-{fix_id}")

            try:
                # Get AKS credentials
                get_credentials_result = self._run_az(
                    [
                        "aks",
                        "get-credentials",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        cluster_name,
                        "--overwrite-existing",
                    ]
                )

                if not get_credentials_result.get(
                    "success", False
                ) and not get_credentials_result.get("simulated", False):
                    return []

                # Use kubectl to get logs
                kubectl_cmd = [
                    "kubectl",
                    "logs",
                    f"deployment/{service_name}-{fix_id}",
                    "--namespace",
                    namespace,
                    f"--tail={kwargs.get('tail', 100)}",
                ]

                container = kwargs.get("container")
                if container:
                    kubectl_cmd.extend(["--container", container])

                try:
                    process = subprocess.run(
                        kubectl_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )

                    if process.returncode != 0:
                        logger.error(
                            f"Error getting Kubernetes logs: {process.stderr.decode()}"
                        )
                        return []

                    logs = process.stdout.decode().splitlines()

                    # Format logs
                    formatted_logs = []
                    for i, log in enumerate(logs):
                        formatted_logs.append(
                            {
                                "timestamp": "2023-01-01T00:00:00Z",  # Placeholder timestamp
                                "message": log,
                                "service_name": service_name,
                                "fix_id": fix_id,
                                "deployment_type": "aks",
                                "index": i,
                            }
                        )

                    return formatted_logs

                except Exception as e:
                    logger.exception(f"Error getting Kubernetes logs: {str(e)}")
                    return []

            except Exception as e:
                logger.exception(f"Error getting Azure AKS deployment logs: {str(e)}")
                return []

        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return []
