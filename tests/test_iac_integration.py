"""
Tests for Infrastructure as Code Integration
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.deployment.multi_environment.hybrid_orchestrator import (
    Environment,
    EnvironmentType,
    HealingContext,
    HealingPlan,
    HealingScope,
    HealingStep,
)
from modules.deployment.multi_environment.iac_integration import (
    ChangeType,
    HelmProvider,
    IaCExecution,
    IaCTool,
    InfrastructureAsCodeIntegration,
    InfrastructureChange,
    ResourceType,
    TerraformProvider,
)


@pytest.fixture
def mock_repositories():
    """Create mock IaC repositories"""
    return [
        {
            "id": "terraform-prod",
            "name": "Production Infrastructure",
            "tool": "terraform",
            "repository_url": "https://github.com/example/infra.git",
            "branch": "main",
            "path": "environments/production",
            "auth": {"token": "mock-token"},
            "auto_apply": False,
            "approval_required": True,
            "metadata": {"tier": "production"},
        },
        {
            "id": "helm-apps",
            "name": "Application Charts",
            "tool": "helm",
            "repository_url": "/local/charts",
            "branch": "main",
            "path": ".",
            "auth": {},
            "auto_apply": True,
            "approval_required": False,
            "metadata": {"type": "applications"},
        },
        {
            "id": "terraform-dev",
            "name": "Development Infrastructure",
            "tool": "terraform",
            "repository_url": "https://github.com/example/infra.git",
            "branch": "develop",
            "path": "environments/development",
            "auth": {"token": "mock-token"},
            "auto_apply": True,
            "approval_required": False,
            "metadata": {"tier": "development"},
        },
    ]


@pytest.fixture
def mock_config(mock_repositories):
    """Create mock configuration"""
    return {"repositories": mock_repositories}


@pytest.fixture
def integration(mock_config):
    """Create IaC integration instance for testing"""
    with patch("modules.deployment.multi_environment.iac_integration.AuditLogger"):
        with patch(
            "modules.deployment.multi_environment.iac_integration.DistributedMonitor"
        ):
            return InfrastructureAsCodeIntegration(mock_config)


@pytest.fixture
def mock_environment():
    """Create mock environment"""
    return Environment(
        id="env-prod-1",
        name="Production US East",
        type=EnvironmentType.CLOUD_AWS,
        region="us-east-1",
        connection_info={},
        capabilities=["auto_scaling"],
        health_status="healthy",
        metadata={
            "iac_repository": {"id": "terraform-prod"},
            "iac_variables": {"instance_type": "t3.large", "cluster_size": 5},
        },
    )


@pytest.mark.asyncio
async def test_terraform_provider_validate():
    """Test Terraform provider validation"""
    provider = TerraformProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Create a simple Terraform file
        tf_content = """
        resource "aws_instance" "test" {
          ami           = "ami-12345678"
          instance_type = "t2.micro"
        }
        """
        (work_dir / "main.tf").write_text(tf_content)

        # Mock terraform command execution
        provider._run_command = AsyncMock(return_value='{"valid": true}')

        valid, errors = await provider.validate(work_dir)
        assert valid is True
        assert errors == []


@pytest.mark.asyncio
async def test_terraform_provider_plan():
    """Test Terraform provider plan generation"""
    provider = TerraformProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Mock terraform plan output
        plan_output = {
            "resource_changes": [
                {
                    "address": "aws_instance.test",
                    "type": "aws_instance",
                    "change": {
                        "actions": ["create"],
                        "before": None,
                        "after": {"ami": "ami-12345678", "instance_type": "t2.micro"},
                        "after_unknown": {},
                        "after_sensitive": {},
                        "replace_paths": [],
                    },
                }
            ]
        }

        provider._run_command = AsyncMock(
            side_effect=["", json.dumps(plan_output)]  # plan command  # show command
        )

        changes = await provider.plan(work_dir, {"environment": "test"})

        assert len(changes) == 1
        assert changes[0].resource_id == "aws_instance.test"
        assert changes[0].change_type == ChangeType.CREATE
        assert changes[0].resource_type == ResourceType.COMPUTE


def test_terraform_resource_type_mapping():
    """Test Terraform resource type mapping"""
    provider = TerraformProvider()

    assert provider._map_terraform_type("aws_instance") == ResourceType.COMPUTE
    assert (
        provider._map_terraform_type("google_compute_instance") == ResourceType.COMPUTE
    )
    assert provider._map_terraform_type("aws_vpc") == ResourceType.NETWORK
    assert provider._map_terraform_type("aws_subnet") == ResourceType.NETWORK
    assert provider._map_terraform_type("aws_s3_bucket") == ResourceType.STORAGE
    assert provider._map_terraform_type("aws_rds_instance") == ResourceType.DATABASE
    assert provider._map_terraform_type("aws_iam_role") == ResourceType.SECURITY
    assert (
        provider._map_terraform_type("aws_lambda_function") == ResourceType.SERVERLESS
    )


def test_infrastructure_change_risk_calculation():
    """Test risk score calculation for infrastructure changes"""
    provider = TerraformProvider()

    # High risk: security resource deletion
    risk = provider._calculate_risk(ResourceType.SECURITY, ChangeType.DELETE)
    assert risk > 0.7

    # Low risk: monitoring resource creation
    risk = provider._calculate_risk(ResourceType.MONITORING, ChangeType.CREATE)
    assert risk < 0.3

    # Medium risk: database update
    risk = provider._calculate_risk(ResourceType.DATABASE, ChangeType.UPDATE)
    assert 0.3 < risk < 0.7


@pytest.mark.asyncio
async def test_helm_provider_validate():
    """Test Helm provider validation"""
    provider = HelmProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Create Chart.yaml
        chart_content = """
        apiVersion: v2
        name: test-chart
        version: 1.0.0
        """
        (work_dir / "Chart.yaml").write_text(chart_content)

        # Mock helm lint
        provider._run_command = AsyncMock(return_value="")

        valid, errors = await provider.validate(work_dir)
        assert valid is True
        assert errors == []


@pytest.mark.asyncio
async def test_helm_provider_apply():
    """Test Helm provider apply"""
    provider = HelmProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Mock helm commands
        provider._check_release_exists = AsyncMock(return_value=False)
        provider._run_command = AsyncMock(return_value="")

        result = await provider.apply(
            work_dir,
            {
                "release_name": "test-release",
                "namespace": "default",
                "values": {"replicas": 3},
            },
        )

        assert result["success"] is True
        assert result["release"] == "test-release"


@pytest.mark.asyncio
async def test_sync_repository_git(integration):
    """Test syncing git repository"""
    repo_id = "terraform-prod"

    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        # Mock successful git clone
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_subprocess.return_value = mock_proc

        # Patch Path.exists to simulate the expected directory structure
        with patch("pathlib.Path.exists") as mock_exists:
            # Make the path check return True for the expected subdirectory
            mock_exists.return_value = True

            success, work_dir = await integration.sync_repository(repo_id)

            assert success is True
        assert work_dir.startswith("/tmp/iac_terraform-prod_") or work_dir.startswith(
            "/var/folders/"
        )  # macOS temp dir

        # Verify git clone was called with correct arguments
        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0]
        assert "git" in call_args
        assert "clone" in call_args


@pytest.mark.asyncio
async def test_sync_repository_local(integration):
    """Test syncing local repository"""
    repo_id = "helm-apps"

    with tempfile.TemporaryDirectory() as local_repo:
        # Update repository URL to temp directory
        integration.repositories[repo_id].repository_url = local_repo

        # Create some files
        Path(local_repo, "Chart.yaml").touch()

        success, work_dir = await integration.sync_repository(repo_id)

        assert success is True
        assert Path(work_dir, "Chart.yaml").exists()


@pytest.mark.asyncio
async def test_validate_infrastructure(integration, mock_environment):
    """Test infrastructure validation"""
    # Mock repository sync
    integration.sync_repository = AsyncMock(return_value=(True, "/tmp/work"))

    # Mock provider validation
    mock_provider = MagicMock()
    mock_provider.validate = AsyncMock(return_value=(True, []))
    integration.providers[IaCTool.TERRAFORM] = mock_provider

    result = await integration.validate_infrastructure(
        "terraform-prod", mock_environment
    )

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["repository"] == "terraform-prod"
    assert result["environment"] == "Production US East"


@pytest.mark.asyncio
async def test_plan_infrastructure_changes(integration, mock_environment):
    """Test planning infrastructure changes"""
    # Mock repository sync
    with tempfile.TemporaryDirectory() as tmpdir:
        integration.sync_repository = AsyncMock(return_value=(True, tmpdir))

        # Mock provider plan
        mock_changes = [
            InfrastructureChange(
                resource_id="aws_instance.web",
                resource_type=ResourceType.COMPUTE,
                change_type=ChangeType.CREATE,
                current_state=None,
                desired_state={"instance_type": "t3.large"},
                impact_analysis={},
                estimated_duration=300,
                risk_score=0.3,
            ),
            InfrastructureChange(
                resource_id="aws_security_group.web",
                resource_type=ResourceType.SECURITY,
                change_type=ChangeType.UPDATE,
                current_state={"ingress": []},
                desired_state={"ingress": [{"port": 443}]},
                impact_analysis={},
                estimated_duration=60,
                risk_score=0.5,
            ),
        ]

        mock_provider = MagicMock()
        mock_provider.plan = AsyncMock(return_value=mock_changes)
        integration.providers[IaCTool.TERRAFORM] = mock_provider

        result = await integration.plan_infrastructure_changes(
            "terraform-prod", mock_environment, {"additional_var": "value"}
        )

        assert result["repository"] == "terraform-prod"
        assert result["environment"] == "Production US East"
        assert len(result["changes"]) == 2
        assert result["summary"]["total_changes"] == 2
        assert result["summary"]["creates"] == 1
        assert result["summary"]["updates"] == 1
        assert result["summary"]["total_risk_score"] == 0.4
        assert result["approval_required"] is True  # Due to repository setting


def test_merge_variables(integration, mock_environment):
    """Test variable merging"""
    custom_vars = {"cluster_size": 10, "new_var": "new_value"}  # Override  # Additional

    merged = integration._merge_variables(mock_environment, custom_vars)

    assert merged["environment"] == "Production US East"
    assert merged["region"] == "us-east-1"
    assert merged["instance_type"] == "t3.large"  # From environment
    assert merged["cluster_size"] == 10  # Overridden
    assert merged["new_var"] == "new_value"  # Added


@pytest.mark.asyncio
async def test_apply_infrastructure_changes(integration, mock_environment):
    """Test applying infrastructure changes"""
    repo_id = "terraform-dev"  # Use dev repo with auto_apply=True

    execution_plan = {
        "summary": {"total_changes": 2, "total_risk_score": 0.2},
        "variables": {"test": "value"},
    }

    # Mock repository sync
    integration.sync_repository = AsyncMock(return_value=(True, "/tmp/work"))

    # Mock provider apply
    mock_provider = MagicMock()
    mock_provider.apply = AsyncMock(
        return_value={"success": True, "resources_created": 1, "resources_updated": 1}
    )
    integration.providers[IaCTool.TERRAFORM] = mock_provider

    # Mock auditor
    integration.auditor.log_event = AsyncMock()

    execution = await integration.apply_infrastructure_changes(
        repo_id, mock_environment, execution_plan, auto_approve=True
    )

    assert execution.status == "completed"
    assert execution.repository.id == repo_id
    assert execution.environment == mock_environment
    assert execution.output["success"] is True
    assert len(execution.errors) == 0

    # Verify audit events
    assert integration.auditor.log_event.call_count == 2


@pytest.mark.asyncio
async def test_handle_infrastructure_healing(integration, mock_environment):
    """Test infrastructure healing coordination"""
    # Create healing context
    healing_context = HealingContext(
        error_id="err-123",
        source_environment=mock_environment,
        affected_environments=[mock_environment],
        scope=HealingScope.ENVIRONMENT,
        dependencies=[],
        constraints={"auto_scale": True, "min_instances": 3, "max_instances": 10},
        priority=1,
        timestamp=datetime.now(timezone.utc),
    )

    healing_plan = HealingPlan(
        plan_id="plan-123",
        context=healing_context,
        steps=[
            HealingStep(
                step_id="scale-1",
                environment=mock_environment,
                action="scale_infrastructure",
                parameters={"infrastructure": {"desired_capacity": 5}},
                dependencies=[],
                timeout=300,
                can_fail=False,
                rollback_action=None,
            )
        ],
        rollback_steps=[],
        approval_required=False,
        estimated_duration=300,
        risk_score=0.3,
    )

    # Mock methods
    integration.validate_infrastructure = AsyncMock(return_value={"valid": True})
    integration.plan_infrastructure_changes = AsyncMock(
        return_value={"summary": {"total_risk_score": 0.2}, "changes": []}
    )
    integration.apply_infrastructure_changes = AsyncMock(
        return_value=IaCExecution(
            execution_id="exec-1",
            repository=integration.repositories["terraform-prod"],
            environment=mock_environment,
            changes=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            status="completed",
            output={"success": True},
            errors=[],
        )
    )

    result = await integration.handle_infrastructure_healing(
        healing_context, healing_plan
    )

    assert result["status"] == "completed"
    assert len(result["executions"]) == 1
    assert result["executions"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_get_infrastructure_state(integration, mock_environment):
    """Test getting infrastructure state"""
    # Mock repository sync
    integration.sync_repository = AsyncMock(return_value=(True, "/tmp/work"))

    # Mock provider get_state
    mock_state = {
        "resources": [{"type": "aws_instance", "name": "web", "instances": [{}]}]
    }
    mock_provider = MagicMock()
    mock_provider.get_state = AsyncMock(return_value=mock_state)
    integration.providers[IaCTool.TERRAFORM] = mock_provider

    result = await integration.get_infrastructure_state(
        "terraform-prod", mock_environment
    )

    assert result["repository"] == "terraform-prod"
    assert result["environment"] == "Production US East"
    assert result["tool"] == "terraform"
    assert "resources" in result["state"]


@pytest.mark.asyncio
async def test_import_existing_infrastructure(integration, mock_environment):
    """Test importing existing infrastructure"""
    # Mock repository sync
    integration.sync_repository = AsyncMock(return_value=(True, "/tmp/work"))

    # Mock provider import
    mock_provider = MagicMock()
    mock_provider.import_resource = AsyncMock(side_effect=[True, False])
    integration.providers[IaCTool.TERRAFORM] = mock_provider

    resources = [
        {"address": "aws_instance.existing", "id": "i-1234567890"},
        {"address": "aws_rds_instance.db", "id": "db-instance-1"},
    ]

    result = await integration.import_existing_infrastructure(
        "terraform-prod", mock_environment, resources
    )

    assert result["repository"] == "terraform-prod"
    assert result["environment"] == "Production US East"
    assert result["imports"]["aws_instance.existing"]["success"] is True
    assert result["imports"]["aws_rds_instance.db"]["success"] is False


def test_change_to_dict(integration):
    """Test converting InfrastructureChange to dictionary"""
    change = InfrastructureChange(
        resource_id="test.resource",
        resource_type=ResourceType.COMPUTE,
        change_type=ChangeType.UPDATE,
        current_state={"old": "value"},
        desired_state={"new": "value"},
        impact_analysis={"dependencies": 2},
        estimated_duration=120,
        risk_score=0.4,
    )

    result = integration._change_to_dict(change)

    assert result["resource_id"] == "test.resource"
    assert result["resource_type"] == "compute"
    assert result["change_type"] == "update"
    assert result["risk_score"] == 0.4


@pytest.mark.asyncio
async def test_list_executions(integration, mock_environment):
    """Test listing IaC executions"""
    # Create some mock executions
    exec1 = IaCExecution(
        execution_id="exec-1",
        repository=integration.repositories["terraform-prod"],
        environment=mock_environment,
        changes=[],
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        status="running",
        output={},
        errors=[],
    )

    exec2 = IaCExecution(
        execution_id="exec-2",
        repository=integration.repositories["terraform-dev"],
        environment=mock_environment,
        changes=[],
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status="completed",
        output={},
        errors=[],
    )

    integration.active_executions = {"exec-1": exec1, "exec-2": exec2}

    # List all executions
    all_execs = await integration.list_executions()
    assert len(all_execs) == 2

    # Filter by environment
    env_execs = await integration.list_executions(environment=mock_environment)
    assert len(env_execs) == 2

    # Filter by status
    running_execs = await integration.list_executions(status="running")
    assert len(running_execs) == 1
    assert running_execs[0].execution_id == "exec-1"
