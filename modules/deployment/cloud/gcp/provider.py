"""
GCP Provider for Homeostasis.

This module provides integration with Google Cloud Platform for deploying fixes.
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

from modules.deployment.cloud.base_provider import BaseCloudProvider
from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class GCPProvider(BaseCloudProvider):
    """
    Google Cloud Platform provider implementation.

    Supports deploying to:
    - Cloud Functions
    - Cloud Run
    - GKE (Google Kubernetes Engine)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize GCP provider.

        Args:
            config: GCP configuration dictionary
        """
        super().__init__(config)

        # GCP-specific configuration
        self.project_id = self.config.get("project_id")
        self.location = self.region or self.config.get("location", "us-central1")
        self.cloud_function_enabled = self.config.get("cloud_function", False)
        self.cloud_run_enabled = self.config.get("cloud_run", False)
        self.gke_enabled = self.config.get("gke_cluster", False)

        # Default deployment type if multiple are enabled
        self.default_deployment_type = None
        if self.cloud_function_enabled:
            self.default_deployment_type = "function"
        elif self.cloud_run_enabled:
            self.default_deployment_type = "run"
        elif self.gke_enabled:
            self.default_deployment_type = "gke"

        # Check if gcloud CLI is available
        self.gcloud_available = self._check_gcloud_available()
        if not self.gcloud_available:
            logger.warning("gcloud CLI not found, GCP operations will be simulated")

    def _check_gcloud_available(self) -> bool:
        """Check if gcloud CLI is available.

        Returns:
            bool: True if gcloud CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "gcloud"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_gcloud(
        self, args: List[str], input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run gcloud CLI command.

        Args:
            args: gcloud CLI arguments
            input_data: Optional input data

        Returns:
            Dict[str, Any]: Command result
        """
        if not self.gcloud_available:
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
                check=False,
            )

            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""

            if process.returncode != 0:
                logger.error(f"gcloud command failed: {stderr}")
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
            logger.exception(f"Error running gcloud command: {str(e)}")
            return {"success": False, "error": str(e)}

    def is_available(self) -> bool:
        """Check if GCP provider is available.

        Returns:
            bool: True if GCP provider is available, False otherwise
        """
        if not self.gcloud_available:
            return False

        if not self.project_id:
            logger.error("GCP project ID not configured")
            return False

        # Try to get project info
        result = self._run_gcloud(["projects", "describe", self.project_id])
        return result.get("success", False)

    def _deploy_function(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to Google Cloud Functions.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")
        entry_point = kwargs.get("entry_point", "main")
        runtime = kwargs.get("runtime", "python39")
        memory = kwargs.get("memory", "256MB")
        timeout = kwargs.get("timeout", "60s")

        if not self.gcloud_available:
            logger.info(f"Simulating GCP Cloud Function deployment: {function_name}")
            return {
                "success": True,
                "simulated": True,
                "function_name": function_name,
                "deployment_type": "function",
            }

        # Check if source path exists
        if not os.path.exists(source_path):
            logger.error(f"Source path not found: {source_path}")
            return {"success": False, "error": f"Source path not found: {source_path}"}

        try:
            # Deploy cloud function
            logger.info(f"Deploying Cloud Function: {function_name}")

            # Build command
            cmd = [
                "functions",
                "deploy",
                function_name,
                "--source",
                source_path,
                "--runtime",
                runtime,
                "--entry-point",
                entry_point,
                "--memory",
                memory,
                "--timeout",
                timeout,
                "--project",
                self.project_id,
                "--region",
                self.location,
            ]

            # Add trigger
            trigger_type = kwargs.get("trigger_type", "http")
            if trigger_type == "http":
                cmd.append("--trigger-http")
            elif trigger_type == "topic":
                topic = kwargs.get("topic")
                if not topic:
                    logger.error("Topic name required for Pub/Sub trigger")
                    return {
                        "success": False,
                        "error": "Topic name required for Pub/Sub trigger",
                    }
                cmd.extend(["--trigger-topic", topic])

            # Set environment variables
            env_vars = kwargs.get("env_vars", {})
            if env_vars:
                env_vars_str = ",".join(
                    [f"{key}={value}" for key, value in env_vars.items()]
                )
                cmd.extend(["--set-env-vars", env_vars_str])

            # Deploy function
            result = self._run_gcloud(cmd)

            # Add tags
            if result.get("success", False) and not result.get("simulated", False):
                self._run_gcloud(
                    [
                        "functions",
                        "add-iam-policy-binding",
                        function_name,
                        "--member",
                        "allUsers",
                        "--role",
                        "roles/cloudfunctions.invoker",
                        "--project",
                        self.project_id,
                        "--region",
                        self.location,
                    ]
                )

            return result

        except Exception as e:
            logger.exception(f"Error deploying to GCP Cloud Function: {str(e)}")
            return {"success": False, "error": str(e)}

    def _deploy_cloud_run(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to Google Cloud Run.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        service_identifier = f"{service_name}-{fix_id}"
        image = kwargs.get("image")

        if not image:
            logger.error("Image required for Cloud Run deployment")
            return {
                "success": False,
                "error": "Image required for Cloud Run deployment",
            }

        if not self.gcloud_available:
            logger.info(f"Simulating GCP Cloud Run deployment: {service_identifier}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_identifier,
                "deployment_type": "run",
            }

        try:
            # Deploy cloud run service
            logger.info(f"Deploying Cloud Run service: {service_identifier}")

            # Build command
            cmd = [
                "run",
                "deploy",
                service_identifier,
                "--image",
                image,
                "--project",
                self.project_id,
                "--region",
                self.location,
                "--platform",
                "managed",
            ]

            # Set memory limit
            memory_limit = kwargs.get("memory", "512Mi")
            cmd.extend(["--memory", memory_limit])

            # Set CPU limit
            cpu_limit = kwargs.get("cpu")
            if cpu_limit:
                cmd.extend(["--cpu", cpu_limit])

            # Set environment variables
            env_vars = kwargs.get("env_vars", {})
            if env_vars:
                env_vars_str = ",".join(
                    [f"{key}={value}" for key, value in env_vars.items()]
                )
                cmd.extend(["--set-env-vars", env_vars_str])

            # Set concurrency
            concurrency = kwargs.get("concurrency")
            if concurrency:
                cmd.extend(["--concurrency", str(concurrency)])

            # Set port
            port = kwargs.get("port")
            if port:
                cmd.extend(["--port", str(port)])

            # Set allow unauthenticated
            allow_unauthenticated = kwargs.get("allow_unauthenticated", True)
            if allow_unauthenticated:
                cmd.append("--allow-unauthenticated")

            # Deploy service
            result = self._run_gcloud(cmd)

            return result

        except Exception as e:
            logger.exception(f"Error deploying to GCP Cloud Run: {str(e)}")
            return {"success": False, "error": str(e)}

    def _deploy_gke(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to Google Kubernetes Engine.

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
            logger.error("Image required for GKE deployment")
            return {"success": False, "error": "Image required for GKE deployment"}

        if not self.gcloud_available:
            logger.info(f"Simulating GCP GKE deployment: {service_identifier}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_identifier,
                "cluster": cluster_name,
                "deployment_type": "gke",
            }

        try:
            # Get GKE credentials
            logger.info(f"Getting GKE credentials for cluster: {cluster_name}")
            get_credentials_result = self._run_gcloud(
                [
                    "container",
                    "clusters",
                    "get-credentials",
                    cluster_name,
                    "--project",
                    self.project_id,
                    "--region",
                    self.location,
                ]
            )

            if not get_credentials_result.get(
                "success", False
            ) and not get_credentials_result.get("simulated", False):
                return get_credentials_result

            # Now we can use kubectl
            from modules.deployment.kubernetes.kubernetes_deployment import \
                KubernetesDeployment

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
                "deployment_type": "gke",
                "kubernetes_result": deploy_result,
            }

        except Exception as e:
            logger.exception(f"Error deploying to GCP GKE: {str(e)}")
            return {"success": False, "error": str(e)}

    def deploy_service(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy a service to GCP.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional provider-specific parameters
                - deployment_type: "function", "run", or "gke"
                - For Cloud Functions: function_name, entry_point, runtime, memory, timeout
                - For Cloud Run: image, memory, cpu, env_vars, concurrency, port
                - For GKE: cluster_name, image, namespace, host

        Returns:
            Dict[str, Any]: Deployment information
        """
        # Check if project ID is set
        if not self.project_id and not kwargs.get("simulated", False):
            logger.error("GCP project ID not configured")
            return {"success": False, "error": "GCP project ID not configured"}

        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        result = None

        if deployment_type == "function" and self.cloud_function_enabled:
            result = self._deploy_function(service_name, fix_id, source_path, **kwargs)
        elif deployment_type == "run" and self.cloud_run_enabled:
            result = self._deploy_cloud_run(service_name, fix_id, source_path, **kwargs)
        elif deployment_type == "gke" and self.gke_enabled:
            result = self._deploy_gke(service_name, fix_id, source_path, **kwargs)
        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return {
                "success": False,
                "error": f"Unsupported or disabled deployment type: {deployment_type}",
            }

        # Log deployment to audit
        try:
            get_audit_logger().log_event(
                event_type="gcp_deployment",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                    "project": self.project_id,
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
        """Undeploy from Google Cloud Functions.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")

        if not self.gcloud_available:
            logger.info(f"Simulating GCP Cloud Function undeployment: {function_name}")
            return {"success": True, "simulated": True, "function_name": function_name}

        # Delete function
        logger.info(f"Deleting Cloud Function: {function_name}")
        return self._run_gcloud(
            [
                "functions",
                "delete",
                function_name,
                "--project",
                self.project_id,
                "--region",
                self.location,
                "--quiet",
            ]
        )

    def _undeploy_cloud_run(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy from Google Cloud Run.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        service_identifier = f"{service_name}-{fix_id}"

        if not self.gcloud_available:
            logger.info(f"Simulating GCP Cloud Run undeployment: {service_identifier}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_identifier,
            }

        # Delete service
        logger.info(f"Deleting Cloud Run service: {service_identifier}")
        return self._run_gcloud(
            [
                "run",
                "services",
                "delete",
                service_identifier,
                "--project",
                self.project_id,
                "--region",
                self.location,
                "--platform",
                "managed",
                "--quiet",
            ]
        )

    def _undeploy_gke(self, service_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Undeploy from Google Kubernetes Engine.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        cluster_name = kwargs.get("cluster_name", "homeostasis")
        namespace = kwargs.get("namespace", "default")

        if not self.gcloud_available:
            logger.info(f"Simulating GCP GKE undeployment: {service_name}-{fix_id}")
            return {
                "success": True,
                "simulated": True,
                "service_name": f"{service_name}-{fix_id}",
                "cluster": cluster_name,
                "namespace": namespace,
            }

        try:
            # Get GKE credentials
            logger.info(f"Getting GKE credentials for cluster: {cluster_name}")
            get_credentials_result = self._run_gcloud(
                [
                    "container",
                    "clusters",
                    "get-credentials",
                    cluster_name,
                    "--project",
                    self.project_id,
                    "--region",
                    self.location,
                ]
            )

            if not get_credentials_result.get(
                "success", False
            ) and not get_credentials_result.get("simulated", False):
                return get_credentials_result

            # Now we can use kubectl
            from modules.deployment.kubernetes.kubernetes_deployment import \
                KubernetesDeployment

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
            logger.exception(f"Error undeploying from GCP GKE: {str(e)}")
            return {"success": False, "error": str(e)}

    def undeploy_service(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy a service from GCP.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters
                - deployment_type: "function", "run", or "gke"
                - For Cloud Functions: function_name
                - For GKE: cluster_name, namespace

        Returns:
            Dict[str, Any]: Undeployment information
        """
        # Check if project ID is set
        if not self.project_id and not kwargs.get("simulated", False):
            logger.error("GCP project ID not configured")
            return {"success": False, "error": "GCP project ID not configured"}

        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        result = None

        if deployment_type == "function" and self.cloud_function_enabled:
            result = self._undeploy_function(service_name, fix_id, **kwargs)
        elif deployment_type == "run" and self.cloud_run_enabled:
            result = self._undeploy_cloud_run(service_name, fix_id, **kwargs)
        elif deployment_type == "gke" and self.gke_enabled:
            result = self._undeploy_gke(service_name, fix_id, **kwargs)
        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return {
                "success": False,
                "error": f"Unsupported or disabled deployment type: {deployment_type}",
            }

        # Log undeployment to audit
        try:
            get_audit_logger().log_event(
                event_type="gcp_undeployment",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                    "project": self.project_id,
                    "location": self.location,
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
                - deployment_type: "function", "run", or "gke"

        Returns:
            Dict[str, Any]: Service status information
        """
        # Check if project ID is set
        if not self.project_id and not kwargs.get("simulated", False):
            logger.error("GCP project ID not configured")
            return {"success": False, "error": "GCP project ID not configured"}

        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        if not self.gcloud_available:
            logger.info(f"Simulating GCP status check for {service_name}-{fix_id}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_name,
                "fix_id": fix_id,
                "deployment_type": deployment_type,
                "status": "ACTIVE",
            }

        if deployment_type == "function" and self.cloud_function_enabled:
            function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")
            logger.info(f"Getting Cloud Function status: {function_name}")

            result = self._run_gcloud(
                [
                    "functions",
                    "describe",
                    function_name,
                    "--project",
                    self.project_id,
                    "--region",
                    self.location,
                ]
            )

            return result

        elif deployment_type == "run" and self.cloud_run_enabled:
            service_identifier = f"{service_name}-{fix_id}"

            logger.info(f"Getting Cloud Run service status: {service_identifier}")

            result = self._run_gcloud(
                [
                    "run",
                    "services",
                    "describe",
                    service_identifier,
                    "--project",
                    self.project_id,
                    "--region",
                    self.location,
                    "--platform",
                    "managed",
                ]
            )

            return result

        elif deployment_type == "gke" and self.gke_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            namespace = kwargs.get("namespace", "default")

            logger.info(f"Getting GKE deployment status: {service_name}-{fix_id}")

            try:
                # Get GKE credentials
                get_credentials_result = self._run_gcloud(
                    [
                        "container",
                        "clusters",
                        "get-credentials",
                        cluster_name,
                        "--project",
                        self.project_id,
                        "--region",
                        self.location,
                    ]
                )

                if not get_credentials_result.get(
                    "success", False
                ) and not get_credentials_result.get("simulated", False):
                    return get_credentials_result

                # Now we can use kubectl
                from modules.deployment.kubernetes.kubernetes_deployment import \
                    KubernetesDeployment

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
                    "deployment_type": "gke",
                    "cluster": cluster_name,
                    "namespace": namespace,
                    "deployment": deployment_status,
                    "service": service_status,
                }

            except Exception as e:
                logger.exception(f"Error getting GKE deployment status: {str(e)}")
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
                - deployment_type: "function", "run", or "gke"
                - limit: Maximum number of log entries to return
                - filter: Filter for log entries

        Returns:
            List[Dict[str, Any]]: Service logs
        """
        # Check if project ID is set
        if not self.project_id and not kwargs.get("simulated", False):
            logger.error("GCP project ID not configured")
            return []

        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        if not self.gcloud_available:
            logger.info(f"Simulating GCP logs retrieval for {service_name}-{fix_id}")
            return [
                {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "message": "Simulated log message",
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                }
            ]

        limit = kwargs.get("limit", 100)
        filter_expr = kwargs.get("filter", "")

        if deployment_type == "function" and self.cloud_function_enabled:
            function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")
            logger.info(f"Getting Cloud Function logs: {function_name}")

            # Build filter
            if not filter_expr:
                filter_expr = f"resource.type=cloud_function AND resource.labels.function_name={function_name}"

            # Get logs
            result = self._run_gcloud(
                [
                    "logging",
                    "read",
                    filter_expr,
                    f"--limit={limit}",
                    "--format=json",
                    "--project",
                    self.project_id,
                ]
            )

            if not result.get("success", False):
                return []

            return result if isinstance(result, list) else []

        elif deployment_type == "run" and self.cloud_run_enabled:
            service_identifier = f"{service_name}-{fix_id}"

            logger.info(f"Getting Cloud Run service logs: {service_identifier}")

            # Build filter
            if not filter_expr:
                filter_expr = f"resource.type=cloud_run_revision AND resource.labels.service_name={service_identifier}"

            # Get logs
            result = self._run_gcloud(
                [
                    "logging",
                    "read",
                    filter_expr,
                    f"--limit={limit}",
                    "--format=json",
                    "--project",
                    self.project_id,
                ]
            )

            if not result.get("success", False):
                return []

            return result if isinstance(result, list) else []

        elif deployment_type == "gke" and self.gke_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            namespace = kwargs.get("namespace", "default")

            logger.info(f"Getting GKE deployment logs: {service_name}-{fix_id}")

            try:
                # Get GKE credentials
                get_credentials_result = self._run_gcloud(
                    [
                        "container",
                        "clusters",
                        "get-credentials",
                        cluster_name,
                        "--project",
                        self.project_id,
                        "--region",
                        self.location,
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
                    f"--tail={limit}",
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
                                "deployment_type": "gke",
                                "index": i,
                            }
                        )

                    return formatted_logs

                except Exception as e:
                    logger.exception(f"Error getting Kubernetes logs: {str(e)}")
                    return []

            except Exception as e:
                logger.exception(f"Error getting GKE deployment logs: {str(e)}")
                return []

        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return []
