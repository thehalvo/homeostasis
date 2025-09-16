"""
Large-Scale Deployment Orchestration Module

Provides orchestration capabilities for enterprise-scale deployments including
Kubernetes, Terraform, Ansible, and other infrastructure automation tools.
"""

import asyncio
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import requests
import yaml

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status"""

    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStrategy(Enum):
    """Deployment strategies"""

    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class ResourceType(Enum):
    """Types of resources that can be deployed"""

    CONTAINER = "container"
    VIRTUAL_MACHINE = "virtual_machine"
    SERVERLESS_FUNCTION = "serverless_function"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    NETWORK = "network"
    STORAGE = "storage"
    CONFIGURATION = "configuration"


@dataclass
class DeploymentResource:
    """Represents a deployable resource"""

    resource_id: str
    name: str
    resource_type: ResourceType
    version: str
    configuration: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentPlan:
    """Represents a deployment plan"""

    plan_id: str
    name: str
    description: str
    strategy: DeploymentStrategy
    resources: List[DeploymentResource] = field(default_factory=list)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    pre_deployment_hooks: List[Dict[str, Any]] = field(default_factory=list)
    post_deployment_hooks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_plan: Optional[Dict[str, Any]] = None
    approval_required: bool = False
    dry_run: bool = False
    timeout: int = 3600  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "strategy": self.strategy.value,
            "resources": [r.__dict__ for r in self.resources],
            "stages": self.stages,
            "pre_deployment_hooks": self.pre_deployment_hooks,
            "post_deployment_hooks": self.post_deployment_hooks,
            "rollback_plan": self.rollback_plan,
            "approval_required": self.approval_required,
            "dry_run": self.dry_run,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""

    deployment_id: str
    plan_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    deployed_resources: List[Dict[str, Any]] = field(default_factory=list)
    failed_resources: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "deployment_id": self.deployment_id,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "deployed_resources": self.deployed_resources,
            "failed_resources": self.failed_resources,
            "logs": self.logs,
            "metrics": self.metrics,
            "errors": self.errors,
        }


class EnterpriseOrchestrator(ABC):
    """Abstract base class for deployment orchestrators"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_parallel_deployments = config.get("max_parallel_deployments", 10)
        self.default_timeout = config.get("default_timeout", 3600)
        self.enable_dry_run = config.get("enable_dry_run", True)
        self._active_deployments: Dict[str, DeploymentResult] = {}

    @abstractmethod
    async def validate_plan(self, plan: DeploymentPlan) -> Tuple[bool, List[str]]:
        """Validate a deployment plan"""
        pass

    @abstractmethod
    async def execute_plan(self, plan: DeploymentPlan) -> DeploymentResult:
        """Execute a deployment plan"""
        pass

    @abstractmethod
    async def get_deployment_status(
        self, deployment_id: str
    ) -> Optional[DeploymentResult]:
        """Get status of a deployment"""
        pass

    @abstractmethod
    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel an active deployment"""
        pass

    @abstractmethod
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment"""
        pass

    @abstractmethod
    async def get_resource_status(self, resource_id: str) -> Dict[str, Any]:
        """Get status of a deployed resource"""
        pass

    async def plan_deployment(
        self, resources: List[DeploymentResource], strategy: DeploymentStrategy
    ) -> DeploymentPlan:
        """Create a deployment plan from resources"""
        plan = DeploymentPlan(
            plan_id=self._generate_plan_id(),
            name=f"Deployment {datetime.utcnow().isoformat()}",
            description="Auto-generated deployment plan",
            strategy=strategy,
            resources=resources,
        )

        # Analyze dependencies and create stages
        stages = self._create_deployment_stages(resources)
        plan.stages = stages

        return plan

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID"""
        import uuid

        return f"plan_{uuid.uuid4().hex[:12]}"

    def _create_deployment_stages(
        self, resources: List[DeploymentResource]
    ) -> List[Dict[str, Any]]:
        """Create deployment stages based on dependencies"""
        stages = []
        deployed: set[str] = set()
        remaining = {r.resource_id: r for r in resources}

        stage_num = 0
        while remaining:
            stage_num += 1
            stage_resources = []

            for resource_id, resource in list(remaining.items()):
                # Check if all dependencies are deployed
                if all(dep in deployed for dep in resource.dependencies):
                    stage_resources.append(resource)
                    del remaining[resource_id]

            if not stage_resources:
                # Circular dependency or missing dependency
                logger.warning(
                    f"Cannot resolve dependencies for resources: {list(remaining.keys())}"
                )
                # Add remaining resources to final stage
                stage_resources = list(remaining.values())
                remaining.clear()

            stages.append(
                {
                    "stage_number": stage_num,
                    "resources": [r.resource_id for r in stage_resources],
                    "parallel": True,  # Resources in same stage can be deployed in parallel
                }
            )

            deployed.update(r.resource_id for r in stage_resources)

        return stages

    async def _execute_health_checks(self, resource: DeploymentResource) -> bool:
        """Execute health checks for a resource"""
        for check in resource.health_checks:
            check_type = check.get("type")

            if check_type == "http":
                if not await self._http_health_check(check):
                    return False
            elif check_type == "tcp":
                if not await self._tcp_health_check(check):
                    return False
            elif check_type == "script":
                if not await self._script_health_check(check):
                    return False

        return True

    async def _http_health_check(self, check: Dict[str, Any]) -> bool:
        """Perform HTTP health check"""
        try:
            url = check.get("url")
            if not url:
                return False
            expected_status = check.get("expected_status", 200)
            timeout = check.get("timeout", 30)

            response = requests.get(url, timeout=timeout)
            return bool(response.status_code == expected_status)

        except Exception as e:
            logger.error(f"HTTP health check failed: {e}")
            return False

    async def _tcp_health_check(self, check: Dict[str, Any]) -> bool:
        """Perform TCP health check"""
        import socket

        try:
            host = check.get("host")
            port = check.get("port")
            timeout = check.get("timeout", 5)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            return result == 0

        except Exception as e:
            logger.error(f"TCP health check failed: {e}")
            return False

    async def _script_health_check(self, check: Dict[str, Any]) -> bool:
        """Perform script-based health check"""
        try:
            script = check.get("script")
            if not script:
                return False
            timeout = check.get("timeout", 60)

            proc = await asyncio.create_subprocess_shell(
                script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            return proc.returncode == 0

        except Exception as e:
            logger.error(f"Script health check failed: {e}")
            return False


class KubernetesOrchestrator(EnterpriseOrchestrator):
    """Kubernetes deployment orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kubeconfig = config.get("kubeconfig")
        self.context = config.get("context")
        self.namespace = config.get("namespace", "default")
        self.kubectl_path = config.get("kubectl_path", "kubectl")

    async def validate_plan(self, plan: DeploymentPlan) -> Tuple[bool, List[str]]:
        """Validate Kubernetes deployment plan"""
        errors = []

        # Check kubectl availability
        try:
            result = await self._run_kubectl(["version", "--client"])
            if result[0] != 0:
                errors.append("kubectl not available or not configured")
        except Exception as e:
            errors.append(f"kubectl validation failed: {e}")

        # Validate each resource
        for resource in plan.resources:
            if resource.resource_type not in [
                ResourceType.CONTAINER,
                ResourceType.CONFIGURATION,
            ]:
                errors.append(
                    f"Resource {resource.name} has unsupported type for Kubernetes: {resource.resource_type}"
                )

            # Validate Kubernetes manifest
            manifest = resource.configuration.get("manifest")
            if not manifest:
                errors.append(f"Resource {resource.name} missing Kubernetes manifest")
            else:
                # Try to parse as YAML
                try:
                    yaml.safe_load(manifest) if isinstance(manifest, str) else manifest
                except Exception as e:
                    errors.append(f"Invalid manifest for {resource.name}: {e}")

        return len(errors) == 0, errors

    async def execute_plan(self, plan: DeploymentPlan) -> DeploymentResult:
        """Execute Kubernetes deployment plan"""
        deployment_id = f"k8s_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        result = DeploymentResult(
            deployment_id=deployment_id,
            plan_id=plan.plan_id,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.utcnow(),
        )

        self._active_deployments[deployment_id] = result

        try:
            # Execute pre-deployment hooks
            for hook in plan.pre_deployment_hooks:
                await self._execute_hook(hook, result)

            # Deploy resources by stage
            for stage in plan.stages:
                stage_tasks = []

                for resource_id in stage["resources"]:
                    resource = next(
                        (r for r in plan.resources if r.resource_id == resource_id),
                        None,
                    )
                    if resource:
                        if stage["parallel"]:
                            task = asyncio.create_task(
                                self._deploy_resource(resource, plan, result)
                            )
                            stage_tasks.append(task)
                        else:
                            await self._deploy_resource(resource, plan, result)

                # Wait for parallel deployments in stage
                if stage_tasks:
                    await asyncio.gather(*stage_tasks, return_exceptions=True)

            # Execute post-deployment hooks
            for hook in plan.post_deployment_hooks:
                await self._execute_hook(hook, result)

            # Final status
            if result.failed_resources:
                result.status = DeploymentStatus.FAILED
            else:
                result.status = DeploymentStatus.COMPLETED

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            result.status = DeploymentStatus.FAILED
            result.errors.append(str(e))

        finally:
            result.end_time = datetime.utcnow()
            del self._active_deployments[deployment_id]

        return result

    async def get_deployment_status(
        self, deployment_id: str
    ) -> Optional[DeploymentResult]:
        """Get Kubernetes deployment status"""
        return self._active_deployments.get(deployment_id)

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel Kubernetes deployment"""
        if deployment_id in self._active_deployments:
            self._active_deployments[deployment_id].status = DeploymentStatus.CANCELLED
            # Additional logic to stop ongoing operations
            return True
        return False

    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback Kubernetes deployment"""
        try:
            # Use kubectl rollout undo
            result = await self._run_kubectl(
                ["rollout", "undo", "deployment", "--all", "-n", self.namespace]
            )

            return result[0] == 0

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def get_resource_status(self, resource_id: str) -> Dict[str, Any]:
        """Get Kubernetes resource status"""
        try:
            # Parse resource ID (format: type/name)
            parts = resource_id.split("/")
            if len(parts) != 2:
                return {"error": "Invalid resource ID format"}

            resource_type, name = parts

            # Get resource details
            result = await self._run_kubectl(
                ["get", resource_type, name, "-n", self.namespace, "-o", "json"]
            )

            if result[0] == 0:
                return cast(Dict[str, Any], json.loads(result[1]))
            else:
                return {"error": result[2]}

        except Exception as e:
            return {"error": str(e)}

    async def _deploy_resource(
        self,
        resource: DeploymentResource,
        plan: DeploymentPlan,
        result: DeploymentResult,
    ):
        """Deploy a single Kubernetes resource"""
        try:
            manifest = resource.configuration.get("manifest")

            # Write manifest to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                if isinstance(manifest, str):
                    f.write(manifest)
                else:
                    yaml.dump(manifest, f)
                temp_file = f.name

            # Apply manifest
            if plan.dry_run:
                kubectl_result = await self._run_kubectl(
                    ["apply", "-f", temp_file, "-n", self.namespace, "--dry-run=client"]
                )
            else:
                kubectl_result = await self._run_kubectl(
                    ["apply", "-f", temp_file, "-n", self.namespace]
                )

            # Clean up temp file
            os.unlink(temp_file)

            if kubectl_result[0] == 0:
                result.deployed_resources.append(
                    {
                        "resource_id": resource.resource_id,
                        "name": resource.name,
                        "type": resource.resource_type.value,
                        "output": kubectl_result[1],
                    }
                )
                result.logs.append(f"Deployed {resource.name}: {kubectl_result[1]}")

                # Wait for resource to be ready and run health checks
                if not plan.dry_run:
                    await self._wait_for_resource(resource)

                    if resource.health_checks:
                        health_ok = await self._execute_health_checks(resource)
                        if not health_ok:
                            raise Exception(f"Health checks failed for {resource.name}")
            else:
                raise Exception(kubectl_result[2])

        except Exception as e:
            logger.error(f"Failed to deploy {resource.name}: {e}")
            result.failed_resources.append(
                {
                    "resource_id": resource.resource_id,
                    "name": resource.name,
                    "error": str(e),
                }
            )
            result.errors.append(f"Failed to deploy {resource.name}: {e}")

    async def _wait_for_resource(self, resource: DeploymentResource):
        """Wait for Kubernetes resource to be ready"""
        manifest = resource.configuration.get("manifest")
        if isinstance(manifest, str):
            manifest_data = yaml.safe_load(manifest)
        else:
            manifest_data = manifest

        kind = manifest_data.get("kind", "").lower()
        name = manifest_data.get("metadata", {}).get("name")

        if kind == "deployment":
            await self._run_kubectl(
                [
                    "wait",
                    "--for=condition=available",
                    f"deployment/{name}",
                    "-n",
                    self.namespace,
                    "--timeout=300s",
                ]
            )
        elif kind == "pod":
            await self._run_kubectl(
                [
                    "wait",
                    "--for=condition=ready",
                    f"pod/{name}",
                    "-n",
                    self.namespace,
                    "--timeout=300s",
                ]
            )
        elif kind == "service":
            # Services are ready immediately
            pass

    async def _execute_hook(self, hook: Dict[str, Any], result: DeploymentResult):
        """Execute deployment hook"""
        try:
            hook_type = hook.get("type")

            if hook_type == "kubectl":
                kubectl_result = await self._run_kubectl(hook.get("command", []))
                result.logs.append(f"Hook output: {kubectl_result[1]}")

            elif hook_type == "script":
                script = hook.get("script")
                if not script:
                    return
                proc = await asyncio.create_subprocess_shell(
                    script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                result.logs.append(f"Hook output: {stdout.decode()}")

        except Exception as e:
            logger.error(f"Hook execution failed: {e}")
            result.errors.append(f"Hook failed: {e}")

    async def _run_kubectl(self, args: List[str]) -> Tuple[int, str, str]:
        """Run kubectl command"""
        cmd = [self.kubectl_path]

        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            cmd.extend(["--context", self.context])

        cmd.extend(args)

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate()

        return proc.returncode or 0, stdout.decode(), stderr.decode()


class TerraformOrchestrator(EnterpriseOrchestrator):
    """Terraform infrastructure orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.terraform_path = config.get("terraform_path", "terraform")
        self.working_directory = config.get("working_directory", ".")
        self.backend_config = config.get("backend_config", {})
        self.variables = config.get("variables", {})
        self.workspace = config.get("workspace", "default")

    async def validate_plan(self, plan: DeploymentPlan) -> Tuple[bool, List[str]]:
        """Validate Terraform deployment plan"""
        errors = []

        # Check terraform availability
        try:
            result = await self._run_terraform(["version"])
            if result[0] != 0:
                errors.append("Terraform not available")
        except Exception as e:
            errors.append(f"Terraform validation failed: {e}")

        # Validate configurations
        for resource in plan.resources:
            config = resource.configuration.get("terraform_config")
            if not config:
                errors.append(
                    f"Resource {resource.name} missing Terraform configuration"
                )

        # Run terraform validate
        if not errors:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write configurations to temp directory
                for resource in plan.resources:
                    config = resource.configuration.get("terraform_config", "")
                    filename = f"{resource.resource_id}.tf"

                    with open(os.path.join(temp_dir, filename), "w") as f:
                        f.write(config)

                # Initialize and validate
                init_result = await self._run_terraform(["init"], cwd=temp_dir)
                if init_result[0] == 0:
                    validate_result = await self._run_terraform(
                        ["validate"], cwd=temp_dir
                    )
                    if validate_result[0] != 0:
                        errors.append(
                            f"Terraform validation failed: {validate_result[2]}"
                        )
                else:
                    errors.append(f"Terraform init failed: {init_result[2]}")

        return len(errors) == 0, errors

    async def execute_plan(self, plan: DeploymentPlan) -> DeploymentResult:
        """Execute Terraform deployment plan"""
        deployment_id = f"tf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        result = DeploymentResult(
            deployment_id=deployment_id,
            plan_id=plan.plan_id,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.utcnow(),
        )

        self._active_deployments[deployment_id] = result

        try:
            with tempfile.TemporaryDirectory() as work_dir:
                # Prepare Terraform workspace
                await self._prepare_workspace(work_dir, plan, result)

                # Initialize Terraform
                init_result = await self._run_terraform(
                    ["init"] + self._get_backend_args(), cwd=work_dir
                )
                if init_result[0] != 0:
                    raise Exception(f"Terraform init failed: {init_result[2]}")

                result.logs.append("Terraform initialized successfully")

                # Select workspace
                if self.workspace != "default":
                    await self._run_terraform(
                        ["workspace", "select", self.workspace], cwd=work_dir
                    )

                # Plan
                plan_file = os.path.join(work_dir, "tfplan")
                plan_cmd = ["plan", "-out", plan_file]

                # Add variables
                for key, value in self.variables.items():
                    plan_cmd.extend(["-var", f"{key}={value}"])

                plan_result = await self._run_terraform(plan_cmd, cwd=work_dir)
                if plan_result[0] != 0:
                    raise Exception(f"Terraform plan failed: {plan_result[2]}")

                result.logs.append("Terraform plan created")
                result.logs.append(plan_result[1])

                # Apply (unless dry run)
                if not plan.dry_run:
                    apply_result = await self._run_terraform(
                        ["apply", "-auto-approve", plan_file], cwd=work_dir
                    )

                    if apply_result[0] == 0:
                        result.logs.append("Terraform apply successful")
                        result.logs.append(apply_result[1])

                        # Get outputs
                        output_result = await self._run_terraform(
                            ["output", "-json"], cwd=work_dir
                        )
                        if output_result[0] == 0:
                            outputs = json.loads(output_result[1])
                            result.metrics["terraform_outputs"] = outputs

                        result.status = DeploymentStatus.COMPLETED
                    else:
                        raise Exception(f"Terraform apply failed: {apply_result[2]}")
                else:
                    result.logs.append("Dry run completed - no changes applied")
                    result.status = DeploymentStatus.COMPLETED

        except Exception as e:
            logger.error(f"Terraform deployment failed: {e}")
            result.status = DeploymentStatus.FAILED
            result.errors.append(str(e))

        finally:
            result.end_time = datetime.utcnow()
            del self._active_deployments[deployment_id]

        return result

    async def get_deployment_status(
        self, deployment_id: str
    ) -> Optional[DeploymentResult]:
        """Get Terraform deployment status"""
        return self._active_deployments.get(deployment_id)

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel Terraform deployment"""
        if deployment_id in self._active_deployments:
            self._active_deployments[deployment_id].status = DeploymentStatus.CANCELLED
            # Terraform doesn't support cancellation during apply
            return True
        return False

    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback Terraform deployment"""
        # Terraform doesn't have built-in rollback
        # Would need to apply previous state or destroy resources
        logger.warning("Terraform rollback not implemented - use state management")
        return False

    async def get_resource_status(self, resource_id: str) -> Dict[str, Any]:
        """Get Terraform resource status"""
        try:
            # Get state
            result = await self._run_terraform(["state", "show", resource_id])

            if result[0] == 0:
                # Parse state output
                return {"state": result[1]}
            else:
                return {"error": result[2]}

        except Exception as e:
            return {"error": str(e)}

    async def _prepare_workspace(
        self, work_dir: str, plan: DeploymentPlan, result: DeploymentResult
    ):
        """Prepare Terraform workspace"""
        # Write resource configurations
        for resource in plan.resources:
            config = resource.configuration.get("terraform_config", "")
            filename = f"{resource.resource_id}.tf"

            filepath = os.path.join(work_dir, filename)
            with open(filepath, "w") as f:
                f.write(config)

            result.logs.append(f"Created {filename}")

        # Write backend configuration if provided
        if self.backend_config:
            backend_tf = (
                'terraform {\n  backend "' + self.backend_config["type"] + '" {\n'
            )
            for key, value in self.backend_config.items():
                if key != "type":
                    backend_tf += f'    {key} = "{value}"\n'
            backend_tf += "  }\n}\n"

            with open(os.path.join(work_dir, "backend.tf"), "w") as f:
                f.write(backend_tf)

    def _get_backend_args(self) -> List[str]:
        """Get backend configuration arguments"""
        args = []
        if self.backend_config:
            for key, value in self.backend_config.items():
                if key != "type":
                    args.extend([f"-backend-config={key}={value}"])
        return args

    async def _run_terraform(
        self, args: List[str], cwd: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """Run terraform command"""
        cmd = [self.terraform_path] + args

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self.working_directory,
        )

        stdout, stderr = await proc.communicate()

        return proc.returncode or 0, stdout.decode(), stderr.decode()


class AnsibleOrchestrator(EnterpriseOrchestrator):
    """Ansible automation orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ansible_path = config.get("ansible_path", "ansible-playbook")
        self.inventory = config.get("inventory", "inventory")
        self.vault_password_file = config.get("vault_password_file")
        self.private_key_file = config.get("private_key_file")
        self.become_password = config.get("become_password")
        self.extra_vars = config.get("extra_vars", {})

    async def validate_plan(self, plan: DeploymentPlan) -> Tuple[bool, List[str]]:
        """Validate Ansible deployment plan"""
        errors = []

        # Check ansible availability
        try:
            result = await self._run_ansible(["--version"])
            if result[0] != 0:
                errors.append("Ansible not available")
        except Exception as e:
            errors.append(f"Ansible validation failed: {e}")

        # Validate playbooks
        for resource in plan.resources:
            playbook = resource.configuration.get("playbook")
            if not playbook:
                errors.append(f"Resource {resource.name} missing Ansible playbook")
            else:
                # Check playbook syntax
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yml", delete=False
                ) as f:
                    f.write(playbook)
                    temp_file = f.name

                syntax_result = await self._run_ansible(["--syntax-check", temp_file])
                os.unlink(temp_file)

                if syntax_result[0] != 0:
                    errors.append(
                        f"Playbook syntax error for {resource.name}: {syntax_result[2]}"
                    )

        return len(errors) == 0, errors

    async def execute_plan(self, plan: DeploymentPlan) -> DeploymentResult:
        """Execute Ansible deployment plan"""
        deployment_id = f"ansible_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        result = DeploymentResult(
            deployment_id=deployment_id,
            plan_id=plan.plan_id,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.utcnow(),
        )

        self._active_deployments[deployment_id] = result

        try:
            # Execute playbooks by stage
            for stage in plan.stages:
                stage_tasks = []

                for resource_id in stage["resources"]:
                    resource = next(
                        (r for r in plan.resources if r.resource_id == resource_id),
                        None,
                    )
                    if resource:
                        if stage["parallel"]:
                            task = asyncio.create_task(
                                self._run_playbook(resource, plan, result)
                            )
                            stage_tasks.append(task)
                        else:
                            await self._run_playbook(resource, plan, result)

                # Wait for parallel playbooks
                if stage_tasks:
                    await asyncio.gather(*stage_tasks, return_exceptions=True)

            # Final status
            if result.failed_resources:
                result.status = DeploymentStatus.FAILED
            else:
                result.status = DeploymentStatus.COMPLETED

        except Exception as e:
            logger.error(f"Ansible deployment failed: {e}")
            result.status = DeploymentStatus.FAILED
            result.errors.append(str(e))

        finally:
            result.end_time = datetime.utcnow()
            del self._active_deployments[deployment_id]

        return result

    async def get_deployment_status(
        self, deployment_id: str
    ) -> Optional[DeploymentResult]:
        """Get Ansible deployment status"""
        return self._active_deployments.get(deployment_id)

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel Ansible deployment"""
        if deployment_id in self._active_deployments:
            self._active_deployments[deployment_id].status = DeploymentStatus.CANCELLED
            # Would need to track and kill ansible processes
            return True
        return False

    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback Ansible deployment"""
        # Would need to run rollback playbooks
        logger.warning("Ansible rollback requires specific rollback playbooks")
        return False

    async def get_resource_status(self, resource_id: str) -> Dict[str, Any]:
        """Get Ansible resource status"""
        # Ansible doesn't maintain state like Terraform
        # Would need to query the actual resources
        return {"message": "Status checking requires target system queries"}

    async def _run_playbook(
        self,
        resource: DeploymentResource,
        plan: DeploymentPlan,
        result: DeploymentResult,
    ):
        """Run an Ansible playbook"""
        try:
            playbook = resource.configuration.get("playbook")
            if not playbook:
                raise ValueError("Playbook content is required")

            # Write playbook to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", delete=False
            ) as f:
                f.write(playbook)
                temp_playbook = f.name

            # Build ansible command
            cmd_args = [temp_playbook]

            # Add inventory
            if self.inventory:
                cmd_args.extend(["-i", self.inventory])

            # Add extra variables
            extra_vars = {
                **self.extra_vars,
                **resource.configuration.get("variables", {}),
            }
            for key, value in extra_vars.items():
                cmd_args.extend(["-e", f"{key}={value}"])

            # Add options
            if plan.dry_run:
                cmd_args.append("--check")

            if self.vault_password_file:
                cmd_args.extend(["--vault-password-file", self.vault_password_file])

            if self.private_key_file:
                cmd_args.extend(["--private-key", self.private_key_file])

            # Run playbook
            ansible_result = await self._run_ansible(cmd_args)

            # Clean up
            os.unlink(temp_playbook)

            if ansible_result[0] == 0:
                result.deployed_resources.append(
                    {
                        "resource_id": resource.resource_id,
                        "name": resource.name,
                        "type": resource.resource_type.value,
                        "output": ansible_result[1],
                    }
                )
                result.logs.append(f"Deployed {resource.name} successfully")
            else:
                raise Exception(ansible_result[2])

        except Exception as e:
            logger.error(f"Failed to run playbook for {resource.name}: {e}")
            result.failed_resources.append(
                {
                    "resource_id": resource.resource_id,
                    "name": resource.name,
                    "error": str(e),
                }
            )
            result.errors.append(f"Failed to deploy {resource.name}: {e}")

    async def _run_ansible(self, args: List[str]) -> Tuple[int, str, str]:
        """Run ansible command"""
        cmd = [self.ansible_path] + args

        env = os.environ.copy()
        if self.become_password:
            env["ANSIBLE_BECOME_PASS"] = self.become_password

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await proc.communicate()

        return proc.returncode or 0, stdout.decode(), stderr.decode()


# Factory function to create orchestrators
def create_orchestrator(
    orchestrator_type: str, config: Dict[str, Any]
) -> Optional[EnterpriseOrchestrator]:
    """Factory function to create orchestrator instances"""
    orchestrators: Dict[str, Type[EnterpriseOrchestrator]] = {
        "kubernetes": KubernetesOrchestrator,
        "k8s": KubernetesOrchestrator,  # Alias
        "terraform": TerraformOrchestrator,
        "ansible": AnsibleOrchestrator,
        # Add more orchestrators as implemented
    }

    orchestrator_class = orchestrators.get(orchestrator_type.lower())
    if not orchestrator_class:
        logger.error(f"Unknown orchestrator type: {orchestrator_type}")
        return None

    try:
        return orchestrator_class(config)
    except Exception as e:
        logger.error(f"Failed to create {orchestrator_type} orchestrator: {e}")
        return None
