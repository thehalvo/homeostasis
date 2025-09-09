"""
Infrastructure as Code Integration

Provides integration with Infrastructure as Code tools like Terraform,
CloudFormation, Pulumi, and Ansible for automated infrastructure healing
and management across environments.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.deployment.multi_environment.config_manager import ChangeAction
from modules.deployment.multi_environment.hybrid_orchestrator import (
    Environment, HealingContext, HealingPlan)
from modules.monitoring.distributed_monitoring import DistributedMonitor
from modules.security.audit import AuditLogger


class IaCTool(Enum):
    """Infrastructure as Code tools"""

    TERRAFORM = "terraform"
    CLOUDFORMATION = "cloudformation"
    PULUMI = "pulumi"
    ANSIBLE = "ansible"
    HELM = "helm"
    KUSTOMIZE = "kustomize"
    CDK = "cdk"
    BICEP = "bicep"


class ResourceType(Enum):
    """Types of infrastructure resources"""

    COMPUTE = "compute"
    NETWORK = "network"
    STORAGE = "storage"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"
    CONTAINER = "container"
    SERVERLESS = "serverless"


class ChangeType(Enum):
    """Types of infrastructure changes"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    REPLACE = "replace"
    NO_CHANGE = "no_change"


@dataclass
class InfrastructureResource:
    """Represents an infrastructure resource"""

    id: str
    name: str
    type: ResourceType
    provider: str  # aws, gcp, azure, etc.
    region: str
    configuration: Dict[str, Any]
    dependencies: List[str]
    tags: Dict[str, str]
    state: str
    metadata: Dict[str, Any]


@dataclass
class InfrastructureChange:
    """Represents a planned infrastructure change"""

    resource_id: str
    resource_type: ResourceType
    change_type: ChangeType
    current_state: Optional[Dict[str, Any]]
    desired_state: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    estimated_duration: int
    risk_score: float


@dataclass
class IaCRepository:
    """Infrastructure as Code repository configuration"""

    id: str
    name: str
    tool: IaCTool
    repository_url: str
    branch: str
    path: str
    auth_config: Dict[str, Any]
    auto_apply: bool
    approval_required: bool
    metadata: Dict[str, Any]


@dataclass
class IaCExecution:
    """Execution context for IaC operations"""

    execution_id: str
    repository: IaCRepository
    environment: Environment
    changes: List[InfrastructureChange]
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    output: Dict[str, Any]
    errors: List[str]


class IaCProvider(ABC):
    """Abstract interface for IaC tool providers"""

    @abstractmethod
    async def validate(self, working_dir: Path) -> Tuple[bool, List[str]]:
        """Validate IaC configuration"""
        pass

    @abstractmethod
    async def plan(
        self, working_dir: Path, variables: Dict[str, Any]
    ) -> List[InfrastructureChange]:
        """Generate execution plan"""
        pass

    @abstractmethod
    async def apply(
        self, working_dir: Path, variables: Dict[str, Any], auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Apply infrastructure changes"""
        pass

    @abstractmethod
    async def destroy(
        self, working_dir: Path, variables: Dict[str, Any], auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Destroy infrastructure"""
        pass

    @abstractmethod
    async def get_state(self, working_dir: Path) -> Dict[str, Any]:
        """Get current infrastructure state"""
        pass

    @abstractmethod
    async def import_resource(
        self, working_dir: Path, resource_addr: str, resource_id: str
    ) -> bool:
        """Import existing resource into state"""
        pass


class TerraformProvider(IaCProvider):
    """Terraform infrastructure provider"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Terraform")
        self.terraform_bin = "terraform"

    async def validate(self, working_dir: Path) -> Tuple[bool, List[str]]:
        """Validate Terraform configuration"""
        try:
            # Initialize Terraform
            await self._run_command(["init", "-backend=false"], working_dir)

            # Validate configuration
            result = await self._run_command(["validate", "-json"], working_dir)
            validation = json.loads(result)

            if validation.get("valid", False):
                return True, []
            else:
                errors = [
                    f"{diag['summary']}: {diag.get('detail', '')}"
                    for diag in validation.get("diagnostics", [])
                ]
                return False, errors

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False, [str(e)]

    async def plan(
        self, working_dir: Path, variables: Dict[str, Any]
    ) -> List[InfrastructureChange]:
        """Generate Terraform plan"""
        try:
            # Build variable arguments
            var_args = []
            for key, value in variables.items():
                var_args.extend(["-var", f"{key}={value}"])

            # Generate plan
            plan_file = working_dir / "tfplan"
            await self._run_command(
                ["plan", "-out", str(plan_file), "-json"] + var_args, working_dir
            )

            # Parse plan JSON
            result = await self._run_command(
                ["show", "-json", str(plan_file)], working_dir
            )
            plan_data = json.loads(result)

            # Convert to InfrastructureChange objects
            changes = []
            for resource_change in plan_data.get("resource_changes", []):
                change = self._parse_resource_change(resource_change)
                if change:
                    changes.append(change)

            return changes

        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}")
            return []

    async def apply(
        self, working_dir: Path, variables: Dict[str, Any], auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Apply Terraform changes"""
        try:
            # Build command
            cmd = ["apply", "-json"]
            if auto_approve:
                cmd.append("-auto-approve")

            # Add variables
            for key, value in variables.items():
                cmd.extend(["-var", f"{key}={value}"])

            # Execute apply
            output = await self._run_command(cmd, working_dir)

            # Parse output
            results = {
                "success": True,
                "resources_created": 0,
                "resources_updated": 0,
                "resources_deleted": 0,
                "outputs": {},
                "raw_output": output,
            }

            # Parse terraform output for resource counts
            if output:
                lines = output.split("\n")
                for line in lines:
                    if "created" in line.lower():
                        results["resources_created"] += 1
                    elif "updated" in line.lower():
                        results["resources_updated"] += 1
                    elif "destroyed" in line.lower():
                        results["resources_deleted"] += 1

            # Get outputs
            outputs_result = await self._run_command(["output", "-json"], working_dir)
            if outputs_result:
                results["outputs"] = json.loads(outputs_result)

            return results

        except Exception as e:
            self.logger.error(f"Apply failed: {e}")
            return {"success": False, "error": str(e)}

    async def destroy(
        self, working_dir: Path, variables: Dict[str, Any], auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Destroy Terraform infrastructure"""
        try:
            cmd = ["destroy", "-json"]
            if auto_approve:
                cmd.append("-auto-approve")

            for key, value in variables.items():
                cmd.extend(["-var", f"{key}={value}"])

            await self._run_command(cmd, working_dir)

            return {"success": True, "message": "Infrastructure destroyed"}

        except Exception as e:
            self.logger.error(f"Destroy failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_state(self, working_dir: Path) -> Dict[str, Any]:
        """Get Terraform state"""
        try:
            result = await self._run_command(["show", "-json"], working_dir)
            return json.loads(result) if result else {}
        except Exception as e:
            self.logger.error(f"Failed to get state: {e}")
            return {}

    async def import_resource(
        self, working_dir: Path, resource_addr: str, resource_id: str
    ) -> bool:
        """Import resource into Terraform state"""
        try:
            await self._run_command(["import", resource_addr, resource_id], working_dir)
            return True
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            return False

    async def _run_command(self, args: List[str], working_dir: Path) -> str:
        """Run Terraform command"""
        cmd = [self.terraform_bin] + args

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise Exception(f"Command failed: {stderr.decode()}")

        return stdout.decode()

    def _parse_resource_change(
        self, resource_change: Dict[str, Any]
    ) -> Optional[InfrastructureChange]:
        """Parse Terraform resource change"""
        change_actions = resource_change.get("change", {}).get("actions", [])
        if not change_actions or change_actions == ["no-op"]:
            return None

        # Map Terraform actions to ChangeType
        if "create" in change_actions:
            change_type = ChangeType.CREATE
        elif "delete" in change_actions:
            change_type = ChangeType.DELETE
        elif "replace" in change_actions:
            change_type = ChangeType.REPLACE
        elif "update" in change_actions:
            change_type = ChangeType.UPDATE
        else:
            change_type = ChangeType.NO_CHANGE

        # Determine resource type
        tf_type = resource_change.get("type", "")
        resource_type = self._map_terraform_type(tf_type)

        return InfrastructureChange(
            resource_id=resource_change.get("address", ""),
            resource_type=resource_type,
            change_type=change_type,
            current_state=resource_change.get("change", {}).get("before"),
            desired_state=resource_change.get("change", {}).get("after"),
            impact_analysis=self._analyze_impact(resource_change),
            estimated_duration=self._estimate_duration(resource_type, change_type),
            risk_score=self._calculate_risk(resource_type, change_type),
        )

    def _map_terraform_type(self, tf_type: str) -> ResourceType:
        """Map Terraform resource type to ResourceType"""
        # Check more specific patterns first to avoid false matches
        if any(t in tf_type for t in ["database", "rds", "sql"]):
            return ResourceType.DATABASE
        elif any(t in tf_type for t in ["vpc", "subnet", "network", "firewall"]):
            return ResourceType.NETWORK
        elif any(t in tf_type for t in ["bucket", "disk", "volume", "storage"]):
            return ResourceType.STORAGE
        elif any(t in tf_type for t in ["security", "iam", "key", "secret"]):
            return ResourceType.SECURITY
        elif any(t in tf_type for t in ["container", "kubernetes", "ecs", "aks"]):
            return ResourceType.CONTAINER
        elif any(t in tf_type for t in ["lambda", "function", "serverless"]):
            return ResourceType.SERVERLESS
        elif any(t in tf_type for t in ["instance", "vm", "compute"]):
            return ResourceType.COMPUTE
        else:
            return ResourceType.MONITORING

    def _analyze_impact(self, resource_change: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of resource change"""
        return {
            "dependencies": len(
                resource_change.get("change", {}).get("after_unknown", {})
            ),
            "sensitive_changes": any(
                resource_change.get("change", {}).get("after_sensitive", {}).values()
            ),
            "force_replacement": resource_change.get("change", {}).get(
                "replace_paths", []
            )
            != [],
        }

    def _estimate_duration(
        self, resource_type: ResourceType, change_type: ChangeType
    ) -> int:
        """Estimate duration for infrastructure change in seconds"""
        base_duration = {
            ResourceType.COMPUTE: 300,
            ResourceType.NETWORK: 120,
            ResourceType.STORAGE: 180,
            ResourceType.DATABASE: 600,
            ResourceType.SECURITY: 60,
            ResourceType.MONITORING: 120,
            ResourceType.CONTAINER: 240,
            ResourceType.SERVERLESS: 60,
        }

        multiplier = {
            ChangeType.CREATE: 1.0,
            ChangeType.UPDATE: 0.5,
            ChangeType.DELETE: 0.3,
            ChangeType.REPLACE: 1.5,
            ChangeType.NO_CHANGE: 0.0,
        }

        return int(
            base_duration.get(resource_type, 120) * multiplier.get(change_type, 1.0)
        )

    def _calculate_risk(
        self, resource_type: ResourceType, change_type: ChangeType
    ) -> float:
        """Calculate risk score for infrastructure change"""
        base_risk = {
            ResourceType.COMPUTE: 0.4,
            ResourceType.NETWORK: 0.6,
            ResourceType.STORAGE: 0.3,
            ResourceType.DATABASE: 0.7,
            ResourceType.SECURITY: 0.8,
            ResourceType.MONITORING: 0.2,
            ResourceType.CONTAINER: 0.5,
            ResourceType.SERVERLESS: 0.3,
        }

        change_multiplier = {
            ChangeType.CREATE: 0.5,
            ChangeType.UPDATE: 0.7,
            ChangeType.DELETE: 0.9,
            ChangeType.REPLACE: 1.2,
            ChangeType.NO_CHANGE: 0.0,
        }

        risk = base_risk.get(resource_type, 0.5) * change_multiplier.get(
            change_type, 1.0
        )
        return min(1.0, risk)


class HelmProvider(IaCProvider):
    """Helm charts infrastructure provider"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Helm")
        self.helm_bin = "helm"

    async def validate(self, working_dir: Path) -> Tuple[bool, List[str]]:
        """Validate Helm chart"""
        try:
            # Lint the chart
            result = await self._run_command(["lint", str(working_dir)])
            # Check if lint passed
            if result and "error" not in result.lower():
                return True, []
            else:
                return False, [result] if result else ["Lint failed"]
        except Exception as e:
            return False, [str(e)]

    async def plan(
        self, working_dir: Path, variables: Dict[str, Any]
    ) -> List[InfrastructureChange]:
        """Generate Helm deployment plan (dry-run)"""
        try:
            # Perform dry-run
            release_name = variables.get("release_name", "default")
            namespace = variables.get("namespace", "default")

            result = await self._run_command(
                [
                    "install",
                    release_name,
                    str(working_dir),
                    "--dry-run",
                    "--debug",
                    "-n",
                    namespace,
                ]
            )

            # Parse dry-run output to identify changes
            changes = []
            if result:
                # Basic parsing of dry-run output
                lines = result.split("\n")
                for line in lines:
                    if "MANIFEST:" in line or "create" in line.lower():
                        changes.append(
                            InfrastructureChange(
                                resource_type="kubernetes",
                                resource_id=release_name,
                                action=ChangeAction.CREATE,
                                before_state={},
                                after_state={"raw": line},
                            )
                        )
            return changes

        except Exception as e:
            self.logger.error(f"Plan failed: {e}")
            return []

    async def apply(
        self, working_dir: Path, variables: Dict[str, Any], auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Apply Helm chart"""
        try:
            release_name = variables.get("release_name", "default")
            namespace = variables.get("namespace", "default")

            # Check if release exists
            existing = await self._check_release_exists(release_name, namespace)

            if existing:
                # Upgrade existing release
                cmd = ["upgrade", release_name, str(working_dir), "-n", namespace]
            else:
                # Install new release
                cmd = ["install", release_name, str(working_dir), "-n", namespace]

            # Add value overrides
            if "values" in variables:
                for key, value in variables["values"].items():
                    cmd.extend(["--set", f"{key}={value}"])

            await self._run_command(cmd)

            return {"success": True, "release": release_name}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def destroy(
        self, working_dir: Path, variables: Dict[str, Any], auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Uninstall Helm release"""
        try:
            release_name = variables.get("release_name", "default")
            namespace = variables.get("namespace", "default")

            await self._run_command(["uninstall", release_name, "-n", namespace])

            return {"success": True, "message": f"Release {release_name} uninstalled"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_state(self, working_dir: Path) -> Dict[str, Any]:
        """Get Helm release state"""
        try:
            # List all releases
            result = await self._run_command(["list", "-A", "-o", "json"])
            releases = json.loads(result) if result else []
            return {"releases": releases}
        except Exception as e:
            return {"error": str(e)}

    async def import_resource(
        self, working_dir: Path, resource_addr: str, resource_id: str
    ) -> bool:
        """Not applicable for Helm"""
        return False

    async def _run_command(self, args: List[str]) -> str:
        """Run Helm command"""
        cmd = [self.helm_bin] + args

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise Exception(f"Command failed: {stderr.decode()}")

        return stdout.decode()

    async def _check_release_exists(self, release_name: str, namespace: str) -> bool:
        """Check if Helm release exists"""
        try:
            await self._run_command(["status", release_name, "-n", namespace])
            return True
        except Exception:
            return False


class InfrastructureAsCodeIntegration:
    """
    Main integration point for Infrastructure as Code tools,
    providing unified interface for infrastructure healing and management.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repositories: Dict[str, IaCRepository] = {}
        self.providers: Dict[IaCTool, IaCProvider] = {
            IaCTool.TERRAFORM: TerraformProvider(),
            IaCTool.HELM: HelmProvider(),
        }
        self.active_executions: Dict[str, IaCExecution] = {}
        self.auditor = AuditLogger()
        self.monitor = DistributedMonitor()
        self.logger = logging.getLogger(__name__)

        # Load repository configurations
        self._load_repositories(config.get("repositories", []))

    def _load_repositories(self, repo_configs: List[Dict[str, Any]]):
        """Load IaC repository configurations"""
        for config in repo_configs:
            repo = IaCRepository(
                id=config["id"],
                name=config["name"],
                tool=IaCTool(config["tool"]),
                repository_url=config["repository_url"],
                branch=config.get("branch", "main"),
                path=config.get("path", "."),
                auth_config=config.get("auth", {}),
                auto_apply=config.get("auto_apply", False),
                approval_required=config.get("approval_required", True),
                metadata=config.get("metadata", {}),
            )
            self.repositories[repo.id] = repo

    async def sync_repository(self, repo_id: str) -> Tuple[bool, str]:
        """Sync IaC repository to local working directory"""
        if repo_id not in self.repositories:
            return False, "Repository not found"

        repo = self.repositories[repo_id]

        try:
            # Create temporary directory
            work_dir = Path(tempfile.mkdtemp(prefix=f"iac_{repo_id}_"))

            # Clone repository
            if repo.repository_url.startswith(("http://", "https://", "git@")):
                await self._clone_git_repo(repo, work_dir)
            else:
                # Local repository
                # Remove the work_dir first since copytree requires destination not to exist
                shutil.rmtree(work_dir)
                shutil.copytree(repo.repository_url, work_dir)

            return True, str(work_dir)

        except Exception as e:
            self.logger.error(f"Failed to sync repository {repo_id}: {e}")
            return False, str(e)

    async def _clone_git_repo(self, repo: IaCRepository, work_dir: Path):
        """Clone git repository"""
        # Build git command
        cmd = ["git", "clone"]

        # Add authentication if needed
        if "token" in repo.auth_config:
            # Use token authentication
            url_parts = repo.repository_url.split("://")
            if len(url_parts) == 2:
                url = f"{url_parts[0]}://oauth2:{repo.auth_config['token']}@{url_parts[1]}"
            else:
                url = repo.repository_url
        else:
            url = repo.repository_url

        cmd.extend(["-b", repo.branch, url, str(work_dir)])

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise Exception(f"Git clone failed: {stderr.decode()}")

        # Navigate to specific path if specified
        if repo.path != ".":
            target_path = work_dir / repo.path
            if not target_path.exists():
                raise Exception(f"Path {repo.path} not found in repository")

    async def validate_infrastructure(
        self, repo_id: str, environment: Environment
    ) -> Dict[str, Any]:
        """Validate infrastructure configuration"""
        # Sync repository
        success, work_dir_str = await self.sync_repository(repo_id)
        if not success:
            return {"valid": False, "errors": [work_dir_str]}

        work_dir = Path(work_dir_str)
        repo = self.repositories[repo_id]

        try:
            # Get provider
            provider = self.providers.get(repo.tool)
            if not provider:
                return {
                    "valid": False,
                    "errors": [f"Provider {repo.tool} not supported"],
                }

            # Validate configuration
            valid, errors = await provider.validate(work_dir)

            return {
                "valid": valid,
                "errors": errors,
                "repository": repo_id,
                "environment": environment.name,
            }

        finally:
            # Cleanup
            if work_dir.exists():
                shutil.rmtree(work_dir)

    async def plan_infrastructure_changes(
        self,
        repo_id: str,
        environment: Environment,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate infrastructure change plan"""
        # Sync repository
        success, work_dir_str = await self.sync_repository(repo_id)
        if not success:
            return {"error": work_dir_str}

        work_dir = Path(work_dir_str)
        repo = self.repositories[repo_id]

        try:
            # Get provider
            provider = self.providers.get(repo.tool)
            if not provider:
                return {"error": f"Provider {repo.tool} not supported"}

            # Merge variables with environment defaults
            all_variables = self._merge_variables(environment, variables or {})

            # Generate plan
            changes = await provider.plan(work_dir, all_variables)

            # Analyze overall impact
            total_risk = (
                sum(c.risk_score for c in changes) / len(changes) if changes else 0
            )
            total_duration = sum(c.estimated_duration for c in changes)

            return {
                "repository": repo_id,
                "environment": environment.name,
                "changes": [self._change_to_dict(c) for c in changes],
                "summary": {
                    "total_changes": len(changes),
                    "creates": sum(
                        1 for c in changes if c.change_type == ChangeType.CREATE
                    ),
                    "updates": sum(
                        1 for c in changes if c.change_type == ChangeType.UPDATE
                    ),
                    "deletes": sum(
                        1 for c in changes if c.change_type == ChangeType.DELETE
                    ),
                    "replaces": sum(
                        1 for c in changes if c.change_type == ChangeType.REPLACE
                    ),
                    "total_risk_score": total_risk,
                    "estimated_duration_seconds": total_duration,
                },
                "approval_required": repo.approval_required or total_risk > 0.7,
            }

        finally:
            # Keep work_dir for potential apply
            pass

    def _merge_variables(
        self, environment: Environment, custom_variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge environment variables with custom variables"""
        # Start with environment defaults
        variables = {
            "environment": environment.name,
            "region": environment.region,
            "environment_id": environment.id,
        }

        # Add environment-specific variables
        if environment.metadata.get("iac_variables"):
            variables.update(environment.metadata["iac_variables"])

        # Override with custom variables
        variables.update(custom_variables)

        return variables

    def _change_to_dict(self, change: InfrastructureChange) -> Dict[str, Any]:
        """Convert InfrastructureChange to dictionary"""
        return {
            "resource_id": change.resource_id,
            "resource_type": change.resource_type.value,
            "change_type": change.change_type.value,
            "impact_analysis": change.impact_analysis,
            "estimated_duration": change.estimated_duration,
            "risk_score": change.risk_score,
        }

    async def apply_infrastructure_changes(
        self,
        repo_id: str,
        environment: Environment,
        execution_plan: Dict[str, Any],
        auto_approve: bool = False,
    ) -> IaCExecution:
        """Apply infrastructure changes"""
        repo = self.repositories.get(repo_id)
        if not repo:
            raise ValueError(f"Repository {repo_id} not found")

        # Check approval requirements
        if not auto_approve and (
            repo.approval_required or execution_plan.get("approval_required", False)
        ):
            approved = await self._get_approval(repo, environment, execution_plan)
            if not approved:
                raise Exception("Changes not approved")

        # Create execution context
        execution = IaCExecution(
            execution_id=f"exec_{datetime.now(timezone.utc).timestamp()}",
            repository=repo,
            environment=environment,
            changes=[],  # Would populate from execution_plan
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            status="running",
            output={},
            errors=[],
        )

        self.active_executions[execution.execution_id] = execution

        # Log execution start
        await self.auditor.log_event(
            "iac_execution_started",
            {
                "execution_id": execution.execution_id,
                "repository": repo_id,
                "environment": environment.name,
                "changes": execution_plan.get("summary", {}),
            },
        )

        try:
            # Sync repository again (in case of changes)
            success, work_dir_str = await self.sync_repository(repo_id)
            if not success:
                raise Exception(work_dir_str)

            work_dir = Path(work_dir_str)

            # Get provider
            provider = self.providers.get(repo.tool)
            if not provider:
                raise Exception(f"Provider {repo.tool} not supported")

            # Apply changes
            variables = self._merge_variables(
                environment, execution_plan.get("variables", {})
            )
            result = await provider.apply(work_dir, variables, auto_approve=True)

            execution.output = result
            execution.status = "completed" if result.get("success", False) else "failed"

            if not result.get("success", False):
                execution.errors.append(result.get("error", "Unknown error"))

        except Exception as e:
            execution.status = "failed"
            execution.errors.append(str(e))
            self.logger.error(f"Execution {execution.execution_id} failed: {e}")

        finally:
            execution.completed_at = datetime.now(timezone.utc)

            # Log execution completion
            await self.auditor.log_event(
                "iac_execution_completed",
                {
                    "execution_id": execution.execution_id,
                    "status": execution.status,
                    "duration": (
                        execution.completed_at - execution.started_at
                    ).total_seconds(),
                    "errors": execution.errors,
                },
            )

            # Cleanup
            if "work_dir" in locals() and work_dir.exists():
                shutil.rmtree(work_dir)

        return execution

    async def _get_approval(
        self,
        repo: IaCRepository,
        environment: Environment,
        execution_plan: Dict[str, Any],
    ) -> bool:
        """Get approval for infrastructure changes"""
        # In practice, would integrate with approval systems
        self.logger.info(
            f"Approval required for {repo.name} in {environment.name}: "
            f"{execution_plan.get('summary', {})}"
        )
        return True  # Auto-approve for now

    async def handle_infrastructure_healing(
        self, healing_context: HealingContext, healing_plan: HealingPlan
    ) -> Dict[str, Any]:
        """Handle infrastructure healing through IaC"""
        results = {"executions": [], "status": "pending"}

        # Find relevant IaC repositories for affected environments
        for env in healing_context.affected_environments:
            # Check if environment has IaC configuration
            iac_config = env.metadata.get("iac_repository")
            if not iac_config:
                continue

            repo_id = iac_config.get("id")
            if repo_id not in self.repositories:
                continue

            repo = self.repositories[repo_id]

            # Determine infrastructure changes needed
            infra_changes = await self._determine_infrastructure_changes(
                healing_context, healing_plan, env
            )

            if not infra_changes:
                continue

            try:
                # Validate infrastructure
                validation = await self.validate_infrastructure(repo_id, env)
                if not validation["valid"]:
                    results["executions"].append(
                        {
                            "environment": env.name,
                            "repository": repo_id,
                            "status": "validation_failed",
                            "errors": validation["errors"],
                        }
                    )
                    continue

                # Plan changes
                plan = await self.plan_infrastructure_changes(
                    repo_id, env, infra_changes
                )

                # Apply if auto-apply is enabled or risk is low
                if (
                    repo.auto_apply
                    or plan.get("summary", {}).get("total_risk_score", 1.0) < 0.3
                ):
                    execution = await self.apply_infrastructure_changes(
                        repo_id, env, plan, auto_approve=repo.auto_apply
                    )

                    results["executions"].append(
                        {
                            "environment": env.name,
                            "repository": repo_id,
                            "execution_id": execution.execution_id,
                            "status": execution.status,
                            "changes_applied": len(execution.changes),
                        }
                    )
                else:
                    results["executions"].append(
                        {
                            "environment": env.name,
                            "repository": repo_id,
                            "status": "pending_approval",
                            "plan": plan,
                        }
                    )

            except Exception as e:
                self.logger.error(f"Infrastructure healing failed for {env.name}: {e}")
                results["executions"].append(
                    {
                        "environment": env.name,
                        "repository": repo_id,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Determine overall status
        if all(e["status"] == "completed" for e in results["executions"]):
            results["status"] = "completed"
        elif any(e["status"] == "failed" for e in results["executions"]):
            results["status"] = "failed"
        elif any(e["status"] == "pending_approval" for e in results["executions"]):
            results["status"] = "pending_approval"

        return results

    async def _determine_infrastructure_changes(
        self,
        healing_context: HealingContext,
        healing_plan: HealingPlan,
        environment: Environment,
    ) -> Dict[str, Any]:
        """Determine infrastructure changes needed for healing"""
        changes = {}

        # Analyze healing steps for infrastructure requirements
        for step in healing_plan.steps:
            if step.environment != environment:
                continue

            # Check if step requires infrastructure changes
            if step.action in [
                "scale_infrastructure",
                "add_resources",
                "modify_network",
            ]:
                if "infrastructure" in step.parameters:
                    changes.update(step.parameters["infrastructure"])

            # Check for auto-scaling requirements
            if healing_context.constraints.get("auto_scale", False):
                changes["auto_scaling_enabled"] = True
                changes["min_size"] = healing_context.constraints.get(
                    "min_instances", 2
                )
                changes["max_size"] = healing_context.constraints.get(
                    "max_instances", 10
                )

        return changes

    async def get_infrastructure_state(
        self, repo_id: str, environment: Environment
    ) -> Dict[str, Any]:
        """Get current infrastructure state"""
        # Sync repository
        success, work_dir_str = await self.sync_repository(repo_id)
        if not success:
            return {"error": work_dir_str}

        work_dir = Path(work_dir_str)
        repo = self.repositories[repo_id]

        try:
            # Get provider
            provider = self.providers.get(repo.tool)
            if not provider:
                return {"error": f"Provider {repo.tool} not supported"}

            # Get state
            state = await provider.get_state(work_dir)

            return {
                "repository": repo_id,
                "environment": environment.name,
                "tool": repo.tool.value,
                "state": state,
            }

        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir)

    async def import_existing_infrastructure(
        self, repo_id: str, environment: Environment, resources: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Import existing infrastructure into IaC management"""
        # Sync repository
        success, work_dir_str = await self.sync_repository(repo_id)
        if not success:
            return {"error": work_dir_str}

        work_dir = Path(work_dir_str)
        repo = self.repositories[repo_id]

        try:
            # Get provider
            provider = self.providers.get(repo.tool)
            if not provider:
                return {"error": f"Provider {repo.tool} not supported"}

            # Import each resource
            results = {}
            for resource in resources:
                resource_addr = resource.get("address")
                resource_id = resource.get("id")

                if not resource_addr or not resource_id:
                    results[resource_addr or "unknown"] = {
                        "success": False,
                        "error": "Missing address or id",
                    }
                    continue

                success = await provider.import_resource(
                    work_dir, resource_addr, resource_id
                )
                results[resource_addr] = {"success": success}

            return {
                "repository": repo_id,
                "environment": environment.name,
                "imports": results,
            }

        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir)

    async def get_execution_status(self, execution_id: str) -> Optional[IaCExecution]:
        """Get status of an IaC execution"""
        return self.active_executions.get(execution_id)

    async def list_executions(
        self, environment: Optional[Environment] = None, status: Optional[str] = None
    ) -> List[IaCExecution]:
        """List IaC executions with optional filters"""
        executions = list(self.active_executions.values())

        if environment:
            executions = [e for e in executions if e.environment.id == environment.id]

        if status:
            executions = [e for e in executions if e.status == status]

        return sorted(executions, key=lambda e: e.started_at, reverse=True)
