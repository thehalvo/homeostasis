"""
AWS Provider for Homeostasis.

This module provides AWS integration for deploying fixes.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from modules.deployment.cloud.base_provider import BaseCloudProvider
from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class AWSProvider(BaseCloudProvider):
    """
    AWS cloud provider implementation.

    Supports deploying to:
    - Lambda functions
    - ECS (Elastic Container Service)
    - EKS (Elastic Kubernetes Service)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize AWS provider.

        Args:
            config: AWS configuration dictionary
        """
        super().__init__(config)

        # AWS-specific configuration
        self.aws_region = self.region or self.config.get("region", "us-west-2")
        self.lambda_enabled = self.config.get("lambda_function", False)
        self.ecs_enabled = self.config.get("ecs_service", False)
        self.eks_enabled = self.config.get("eks_cluster", False)

        # Default deployment type if multiple are enabled
        self.default_deployment_type = None
        if self.lambda_enabled:
            self.default_deployment_type = "lambda"
        elif self.ecs_enabled:
            self.default_deployment_type = "ecs"
        elif self.eks_enabled:
            self.default_deployment_type = "eks"

        # Check if AWS CLI is available
        self.aws_cli_available = self._check_aws_cli_available()
        if not self.aws_cli_available:
            logger.warning("AWS CLI not found, AWS operations will be simulated")

    def _check_aws_cli_available(self) -> bool:
        """Check if AWS CLI is available.

        Returns:
            bool: True if AWS CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "aws"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_aws_cli(
        self, args: List[str], input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run AWS CLI command.

        Args:
            args: AWS CLI arguments
            input_data: Optional input data

        Returns:
            Dict[str, Any]: Command result
        """
        if not self.aws_cli_available:
            logger.info(f"Simulating AWS CLI command: aws {' '.join(args)}")
            return {"success": True, "simulated": True}

        try:
            cmd = ["aws", "--region", self.aws_region] + args
            logger.debug(f"Running AWS CLI command: {' '.join(cmd)}")

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
                logger.error(f"AWS CLI command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }

            # Try to parse JSON output if possible
            try:
                if (stdout and
                        (stdout.strip().startswith("{") or
                         stdout.strip().startswith("[")):
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

    def is_available(self) -> bool:
        """Check if AWS provider is available.

        Returns:
            bool: True if AWS provider is available, False otherwise
        """
        if not self.aws_cli_available:
            return False

        # Try to get caller identity
        result = self._run_aws_cli(["sts", "get-caller-identity"])
        return result.get("success", False)

    def _deploy_lambda(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to AWS Lambda.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")
        handler = kwargs.get("handler", "app.lambda_handler")
        runtime = kwargs.get("runtime", "python3.8")
        memory_size = kwargs.get("memory_size", 128)
        timeout = kwargs.get("timeout", 30)

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS Lambda deployment: {function_name}")
            return {
                "success": True,
                "simulated": True,
                "function_name": function_name,
                "deployment_type": "lambda",
            }

        # Check if source path exists
        if not os.path.exists(source_path):
            logger.error(f"Source path not found: {source_path}")
            return {"success": False, "error": f"Source path not found: {source_path}"}

        try:
            # Create a temporary zip file
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
                zip_path = temp_file.name

            # Create a zip file
            zip_cmd = ["zip", "-r", zip_path, "."]
            zip_process = subprocess.run(
                zip_cmd,
                cwd=source_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if zip_process.returncode != 0:
                logger.error(
                    f"Failed to create zip file: {zip_process.stderr.decode()}"
                )
                return {"success": False, "error": "Failed to create zip file"}

            # Check if function exists
            get_function_result = self._run_aws_cli(
                ["lambda", "get-function", "--function-name", function_name]
            )

            if get_function_result.get(
                "success", False
            ) and not get_function_result.get("simulated", False):
                # Update existing function
                logger.info(f"Updating existing Lambda function: {function_name}")

                # Update function code
                update_code_result = self._run_aws_cli(
                    [
                        "lambda",
                        "update-function-code",
                        "--function-name",
                        function_name,
                        "--zip-file",
                        f"fileb://{zip_path}",
                    ]
                )

                if not update_code_result.get("success", False):
                    return update_code_result

                # Update function configuration
                update_config_result = self._run_aws_cli(
                    [
                        "lambda",
                        "update-function-configuration",
                        "--function-name",
                        function_name,
                        "--handler",
                        handler,
                        "--runtime",
                        runtime,
                        "--memory-size",
                        str(memory_size),
                        "--timeout",
                        str(timeout),
                    ]
                )

                return update_config_result

            else:
                # Create new function
                logger.info(f"Creating new Lambda function: {function_name}")

                # Create execution role if needed
                role_name = kwargs.get("role_name", "homeostasis-lambda-role")
                role_arn = kwargs.get("role_arn")

                if not role_arn:
                    # Get or create role
                    get_role_result = self._run_aws_cli(
                        ["iam", "get-role", "--role-name", role_name]
                    )

                    if get_role_result.get(
                        "success", False
                    ) and not get_role_result.get("simulated", False):
                        role_arn = get_role_result.get("Role", {}).get("Arn")
                    else:
                        # Create role with basic execution policy
                        logger.info(f"Creating Lambda execution role: {role_name}")

                        # Create assume role policy document
                        assume_role_policy = {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Principal": {"Service": "lambda.amazonaws.com"},
                                    "Action": "sts:AssumeRole",
                                }
                            ],
                        }

                        create_role_result = self._run_aws_cli(
                            [
                                "iam",
                                "create-role",
                                "--role-name",
                                role_name,
                                "--assume-role-policy-document",
                                json.dumps(assume_role_policy),
                            ]
                        )

                        if not create_role_result.get("success", False):
                            return create_role_result

                        role_arn = create_role_result.get("Role", {}).get("Arn")

                        # Attach basic execution policy
                        attach_policy_result = self._run_aws_cli(
                            [
                                "iam",
                                "attach-role-policy",
                                "--role-name",
                                role_name,
                                "--policy-arn",
                                "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                            ]
                        )

                        if not attach_policy_result.get("success", False):
                            return attach_policy_result

                        # Wait for role to propagate
                        logger.info("Waiting for role to propagate...")
                        import time

                        time.sleep(10)

                # Create function
                create_function_result = self._run_aws_cli(
                    [
                        "lambda",
                        "create-function",
                        "--function-name",
                        function_name,
                        "--runtime",
                        runtime,
                        "--role",
                        role_arn,
                        "--handler",
                        handler,
                        "--zip-file",
                        f"fileb://{zip_path}",
                        "--memory-size",
                        str(memory_size),
                        "--timeout",
                        str(timeout),
                        "--tags",
                        f"Service={service_name},FixId={fix_id}",
                    ]
                )

                return create_function_result

        except Exception as e:
            logger.exception(f"Error deploying to AWS Lambda: {str(e)}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up temporary zip file
            if "zip_path" in locals() and os.path.exists(zip_path):
                os.unlink(zip_path)

    def _deploy_ecs(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to AWS ECS.

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
        container_port = kwargs.get("container_port", 8000)
        cpu = kwargs.get("cpu", "256")
        memory = kwargs.get("memory", "512")

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS ECS deployment: {service_identifier}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_identifier,
                "cluster": cluster_name,
                "deployment_type": "ecs",
            }

        try:
            # Check if cluster exists
            get_cluster_result = self._run_aws_cli(
                ["ecs", "describe-clusters", "--clusters", cluster_name]
            )

            if get_cluster_result.get("success", False) and not get_cluster_result.get(
                "simulated", False
            ):
                clusters = get_cluster_result.get("clusters", [])
                if not clusters:
                    # Create cluster
                    logger.info(f"Creating ECS cluster: {cluster_name}")
                    create_cluster_result = self._run_aws_cli(
                        ["ecs", "create-cluster", "--cluster-name", cluster_name]
                    )

                    if not create_cluster_result.get("success", False):
                        return create_cluster_result

            # Create task definition JSON
            task_def = {
                "family": service_identifier,
                "networkMode": "awsvpc",
                "executionRoleArn": kwargs.get(
                    "execution_role_arn", "ecsTaskExecutionRole"
                ),
                "containerDefinitions": [
                    {
                        "name": service_identifier,
                        "image": image,
                        "essential": True,
                        "portMappings": [
                            {
                                "containerPort": container_port,
                                "hostPort": container_port,
                                "protocol": "tcp",
                            }
                        ],
                        "logConfiguration": {
                            "logDriver": "awslogs",
                            "options": {
                                "awslogs-group": f"/ecs/{service_identifier}",
                                "awslogs-region": self.aws_region,
                                "awslogs-stream-prefix": "ecs",
                            },
                        },
                    }
                ],
                "requiresCompatibilities": ["FARGATE"],
                "cpu": cpu,
                "memory": memory,
                "tags": [
                    {"key": "Service", "value": service_name},
                    {"key": "FixId", "value": fix_id},
                ],
            }

            # Register task definition
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_file:
                json.dump(task_def, temp_file)
                task_def_path = temp_file.name

            register_task_result = self._run_aws_cli(
                [
                    "ecs",
                    "register-task-definition",
                    "--cli-input-json",
                    f"file://{task_def_path}",
                ]
            )

            if not register_task_result.get("success", False):
                return register_task_result

            task_definition_arn = register_task_result.get("taskDefinition", {}).get(
                "taskDefinitionArn"
            )

            # Check if service exists
            get_service_result = self._run_aws_cli(
                [
                    "ecs",
                    "describe-services",
                    "--cluster",
                    cluster_name,
                    "--services",
                    service_identifier,
                ]
            )

            services = get_service_result.get("services", [])

            if (services and
                    services[0].get("status") != "INACTIVE" and
                    not get_service_result.get("simulated", False)):
                # Update existing service
                logger.info(f"Updating existing ECS service: {service_identifier}")
                update_service_result = self._run_aws_cli(
                    [
                        "ecs",
                        "update-service",
                        "--cluster",
                        cluster_name,
                        "--service",
                        service_identifier,
                        "--task-definition",
                        task_definition_arn,
                        "--force-new-deployment",
                    ]
                )

                return update_service_result

            else:
                # Create new service
                logger.info(f"Creating new ECS service: {service_identifier}")

                # Get subnet IDs
                subnets = kwargs.get("subnets", [])
                security_groups = kwargs.get("security_groups", [])

                # Create service
                create_service_result = self._run_aws_cli(
                    [
                        "ecs",
                        "create-service",
                        "--cluster",
                        cluster_name,
                        "--service-name",
                        service_identifier,
                        "--task-definition",
                        task_definition_arn,
                        "--desired-count",
                        "1",
                        "--launch-type",
                        "FARGATE",
                        "--network-configuration",
                        json.dumps(
                            {
                                "awsvpcConfiguration": {
                                    "subnets": subnets,
                                    "securityGroups": security_groups,
                                    "assignPublicIp": "ENABLED",
                                }
                            }
                        ),
                    ]
                )

                return create_service_result

        except Exception as e:
            logger.exception(f"Error deploying to AWS ECS: {str(e)}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up temporary files
            if "task_def_path" in locals() and os.path.exists(task_def_path):
                os.unlink(task_def_path)

    def _deploy_eks(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy to AWS EKS.

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

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS EKS deployment: {service_identifier}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_identifier,
                "cluster": cluster_name,
                "deployment_type": "eks",
            }

        try:
            # Check if eksctl is available
            eksctl_available = False
            try:
                result = subprocess.run(
                    ["which", "eksctl"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                eksctl_available = result.returncode == 0
            except Exception:
                pass

            if not eksctl_available:
                logger.warning("eksctl not found, EKS operations will be simulated")
                return {
                    "success": True,
                    "simulated": True,
                    "service_name": service_identifier,
                    "cluster": cluster_name,
                    "deployment_type": "eks",
                }

            # Update kubeconfig
            update_kubeconfig_result = self._run_aws_cli(
                [
                    "eks",
                    "update-kubeconfig",
                    "--name",
                    cluster_name,
                    "--region",
                    self.aws_region,
                ]
            )

            if not update_kubeconfig_result.get("success", False):
                return update_kubeconfig_result

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
                "deployment_type": "eks",
                "kubernetes_result": deploy_result,
            }

        except Exception as e:
            logger.exception(f"Error deploying to AWS EKS: {str(e)}")
            return {"success": False, "error": str(e)}

    def deploy_service(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy a service to AWS.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional provider-specific parameters
                - deployment_type: "lambda", "ecs", or "eks"
                - For Lambda: function_name, handler, runtime, memory_size, timeout
                - For ECS: cluster_name, image, container_port, cpu, memory
                - For EKS: cluster_name, image, namespace, host

        Returns:
            Dict[str, Any]: Deployment information
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        result = None

        if deployment_type == "lambda" and self.lambda_enabled:
            result = self._deploy_lambda(service_name, fix_id, source_path, **kwargs)
        elif deployment_type == "ecs" and self.ecs_enabled:
            result = self._deploy_ecs(service_name, fix_id, source_path, **kwargs)
        elif deployment_type == "eks" and self.eks_enabled:
            result = self._deploy_eks(service_name, fix_id, source_path, **kwargs)
        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return {
                "success": False,
                "error": f"Unsupported or disabled deployment type: {deployment_type}",
            }

        # Log deployment to audit
        try:
            get_audit_logger().log_event(
                event_type="aws_deployment",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                    "region": self.aws_region,
                    "success": result.get("success", False),
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        return result

    def _undeploy_lambda(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy from AWS Lambda.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS Lambda undeployment: {function_name}")
            return {"success": True, "simulated": True, "function_name": function_name}

        # Delete function
        logger.info(f"Deleting Lambda function: {function_name}")
        return self._run_aws_cli(
            ["lambda", "delete-function", "--function-name", function_name]
        )

    def _undeploy_ecs(self, service_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Undeploy from AWS ECS.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        cluster_name = kwargs.get("cluster_name", "homeostasis")
        service_identifier = f"{service_name}-{fix_id}"

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS ECS undeployment: {service_identifier}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_identifier,
                "cluster": cluster_name,
            }

        # Update service to 0 desired count
        logger.info(f"Scaling ECS service to 0: {service_identifier}")
        update_result = self._run_aws_cli(
            [
                "ecs",
                "update-service",
                "--cluster",
                cluster_name,
                "--service",
                service_identifier,
                "--desired-count",
                "0",
            ]
        )

        if not update_result.get("success", False) and not update_result.get(
            "simulated", False
        ):
            return update_result

        # Delete service
        logger.info(f"Deleting ECS service: {service_identifier}")
        delete_result = self._run_aws_cli(
            [
                "ecs",
                "delete-service",
                "--cluster",
                cluster_name,
                "--service",
                service_identifier,
                "--force",
            ]
        )

        if not delete_result.get("success", False) and not delete_result.get(
            "simulated", False
        ):
            return delete_result

        # Deregister task definition
        logger.info(f"Deregistering ECS task definition: {service_identifier}")
        list_result = self._run_aws_cli(
            [
                "ecs",
                "list-task-definitions",
                "--family-prefix",
                service_identifier,
                "--sort",
                "DESC",
                "--max-items",
                "1",
            ]
        )

        if not list_result.get("success", False) and not list_result.get(
            "simulated", False
        ):
            return list_result

        task_definitions = list_result.get("taskDefinitionArns", [])
        if task_definitions:
            logger.info(f"Deregistering task definition: {task_definitions[0]}")
            deregister_result = self._run_aws_cli(
                [
                    "ecs",
                    "deregister-task-definition",
                    "--task-definition",
                    task_definitions[0].split("/")[1],
                ]
            )

            return deregister_result

        return {"success": True}

    def _undeploy_eks(self, service_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Undeploy from AWS EKS.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Undeployment information
        """
        cluster_name = kwargs.get("cluster_name", "homeostasis")
        namespace = kwargs.get("namespace", "default")

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS EKS undeployment: {service_name}-{fix_id}")
            return {
                "success": True,
                "simulated": True,
                "service_name": f"{service_name}-{fix_id}",
                "cluster": cluster_name,
                "namespace": namespace,
            }

        try:
            # Check if eksctl is available
            eksctl_available = False
            try:
                result = subprocess.run(
                    ["which", "eksctl"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                eksctl_available = result.returncode == 0
            except Exception:
                pass

            if not eksctl_available:
                logger.warning("eksctl not found, EKS operations will be simulated")
                return {
                    "success": True,
                    "simulated": True,
                    "service_name": f"{service_name}-{fix_id}",
                    "cluster": cluster_name,
                    "namespace": namespace,
                }

            # Update kubeconfig
            update_kubeconfig_result = self._run_aws_cli(
                [
                    "eks",
                    "update-kubeconfig",
                    "--name",
                    cluster_name,
                    "--region",
                    self.aws_region,
                ]
            )

            if not update_kubeconfig_result.get("success", False):
                return update_kubeconfig_result

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
            logger.exception(f"Error undeploying from AWS EKS: {str(e)}")
            return {"success": False, "error": str(e)}

    def undeploy_service(
        self, service_name: str, fix_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Undeploy a service from AWS.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            **kwargs: Additional provider-specific parameters
                - deployment_type: "lambda", "ecs", or "eks"
                - For Lambda: function_name
                - For ECS: cluster_name
                - For EKS: cluster_name, namespace

        Returns:
            Dict[str, Any]: Undeployment information
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        result = None

        if deployment_type == "lambda" and self.lambda_enabled:
            result = self._undeploy_lambda(service_name, fix_id, **kwargs)
        elif deployment_type == "ecs" and self.ecs_enabled:
            result = self._undeploy_ecs(service_name, fix_id, **kwargs)
        elif deployment_type == "eks" and self.eks_enabled:
            result = self._undeploy_eks(service_name, fix_id, **kwargs)
        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return {
                "success": False,
                "error": f"Unsupported or disabled deployment type: {deployment_type}",
            }

        # Log undeployment to audit
        try:
            get_audit_logger().log_event(
                event_type="aws_undeployment",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                    "region": self.aws_region,
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
                - deployment_type: "lambda", "ecs", or "eks"

        Returns:
            Dict[str, Any]: Service status information
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS status check for {service_name}-{fix_id}")
            return {
                "success": True,
                "simulated": True,
                "service_name": service_name,
                "fix_id": fix_id,
                "deployment_type": deployment_type,
                "status": "ACTIVE",
            }

        if deployment_type == "lambda" and self.lambda_enabled:
            function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")
            logger.info(f"Getting Lambda function status: {function_name}")

            result = self._run_aws_cli(
                ["lambda", "get-function", "--function-name", function_name]
            )

            return result

        elif deployment_type == "ecs" and self.ecs_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            service_identifier = f"{service_name}-{fix_id}"

            logger.info(f"Getting ECS service status: {service_identifier}")

            result = self._run_aws_cli(
                [
                    "ecs",
                    "describe-services",
                    "--cluster",
                    cluster_name,
                    "--services",
                    service_identifier,
                ]
            )

            return result

        elif deployment_type == "eks" and self.eks_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            namespace = kwargs.get("namespace", "default")

            logger.info(f"Getting EKS deployment status: {service_name}-{fix_id}")

            try:
                # Update kubeconfig
                update_kubeconfig_result = self._run_aws_cli(
                    [
                        "eks",
                        "update-kubeconfig",
                        "--name",
                        cluster_name,
                        "--region",
                        self.aws_region,
                    ]
                )

                if not update_kubeconfig_result.get("success", False):
                    return update_kubeconfig_result

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
                    "deployment_type": "eks",
                    "cluster": cluster_name,
                    "namespace": namespace,
                    "deployment": deployment_status,
                    "service": service_status,
                }

            except Exception as e:
                logger.exception(f"Error getting EKS deployment status: {str(e)}")
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
                - deployment_type: "lambda", "ecs", or "eks"
                - For Lambda: function_name, start_time, end_time
                - For ECS: cluster_name, start_time, end_time
                - For EKS: cluster_name, namespace, container, tail

        Returns:
            List[Dict[str, Any]]: Service logs
        """
        # Determine deployment type
        deployment_type = kwargs.get("deployment_type", self.default_deployment_type)

        if not self.aws_cli_available:
            logger.info(f"Simulating AWS logs retrieval for {service_name}-{fix_id}")
            return [
                {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "message": "Simulated log message",
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "deployment_type": deployment_type,
                }
            ]

        if deployment_type == "lambda" and self.lambda_enabled:
            function_name = kwargs.get("function_name", f"{service_name}-{fix_id}")
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")

            logger.info(f"Getting Lambda function logs: {function_name}")

            # Build command
            cmd = [
                "logs",
                "filter-log-events",
                "--log-group-name",
                f"/aws/lambda/{function_name}",
            ]

            if start_time:
                cmd.extend(["--start-time", str(start_time)])

            if end_time:
                cmd.extend(["--end-time", str(end_time)])

            result = self._run_aws_cli(cmd)

            if not result.get("success", False):
                return []

            return result.get("events", [])

        elif deployment_type == "ecs" and self.ecs_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            service_identifier = f"{service_name}-{fix_id}"
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")

            logger.info(f"Getting ECS service logs: {service_identifier}")

            # Build command
            cmd = [
                "logs",
                "filter-log-events",
                "--log-group-name",
                f"/ecs/{service_identifier}",
            ]

            if start_time:
                cmd.extend(["--start-time", str(start_time)])

            if end_time:
                cmd.extend(["--end-time", str(end_time)])

            result = self._run_aws_cli(cmd)

            if not result.get("success", False):
                return []

            return result.get("events", [])

        elif deployment_type == "eks" and self.eks_enabled:
            cluster_name = kwargs.get("cluster_name", "homeostasis")
            namespace = kwargs.get("namespace", "default")
            container = kwargs.get("container")
            tail = kwargs.get("tail", 100)

            logger.info(f"Getting EKS deployment logs: {service_name}-{fix_id}")

            try:
                # Update kubeconfig
                update_kubeconfig_result = self._run_aws_cli(
                    [
                        "eks",
                        "update-kubeconfig",
                        "--name",
                        cluster_name,
                        "--region",
                        self.aws_region,
                    ]
                )

                if not update_kubeconfig_result.get("success", False):
                    return []

                # Use kubectl to get logs
                kubectl_cmd = [
                    "kubectl",
                    "logs",
                    f"deployment/{service_name}-{fix_id}",
                    "--namespace",
                    namespace,
                    f"--tail={tail}",
                ]

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
                                "deployment_type": "eks",
                                "index": i,
                            }
                        )

                    return formatted_logs

                except Exception as e:
                    logger.exception(f"Error getting Kubernetes logs: {str(e)}")
                    return []

            except Exception as e:
                logger.exception(f"Error getting EKS deployment logs: {str(e)}")
                return []

        else:
            logger.error(f"Unsupported or disabled deployment type: {deployment_type}")
            return []
