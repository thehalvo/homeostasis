"""
Tests for Hybrid Cloud/On-Premise Healing Coordination
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from modules.deployment.multi_environment.hybrid_orchestrator import (
    HybridCloudOrchestrator,
    Environment,
    EnvironmentType,
    HealingContext,
    HealingScope,
    HealingPlan,
    HealingStep,
    CloudConnector,
    OnPremiseConnector
)


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "environments": [
            {
                "id": "prod-aws",
                "name": "Production AWS",
                "type": "cloud_aws",
                "region": "us-east-1",
                "connection": {
                    "access_key": "mock_key",
                    "secret_key": "mock_secret"
                },
                "capabilities": ["auto_scaling", "load_balancing"],
                "metadata": {
                    "tags": ["production"],
                    "dependency_level": 1
                }
            },
            {
                "id": "prod-onprem",
                "name": "Production On-Premise",
                "type": "on_premise",
                "connection": {
                    "host": "192.168.1.100",
                    "port": 22
                },
                "capabilities": ["high_performance"],
                "metadata": {
                    "tags": ["production"],
                    "dependency_level": 0
                }
            },
            {
                "id": "dev-gcp",
                "name": "Development GCP",
                "type": "cloud_gcp",
                "region": "us-central1",
                "connection": {
                    "project_id": "test-project",
                    "credentials": "mock_creds"
                },
                "capabilities": ["auto_scaling"],
                "metadata": {
                    "tags": ["development"],
                    "dependency_level": 2
                }
            }
        ]
    }


@pytest.fixture
def orchestrator(mock_config):
    """Create orchestrator instance for testing"""
    with patch('modules.deployment.multi_environment.hybrid_orchestrator.DistributedMonitor'):
        with patch('modules.deployment.multi_environment.hybrid_orchestrator.SecurityAuditor'):
            return HybridCloudOrchestrator(mock_config)


@pytest.mark.asyncio
async def test_initialization(orchestrator):
    """Test orchestrator initialization"""
    # Mock connector methods
    for connector in orchestrator.connectors.values():
        connector.connect = AsyncMock(return_value=True)
        connector.get_health_status = AsyncMock(return_value={"status": "healthy"})
    
    result = await orchestrator.initialize()
    assert result is True
    
    # Verify all environments were initialized
    assert len(orchestrator.environments) == 3
    assert all(env.health_status == "healthy" for env in orchestrator.environments.values())


@pytest.mark.asyncio
async def test_detect_cross_environment_issue(orchestrator):
    """Test detection of cross-environment issues"""
    # Initialize environments
    for connector in orchestrator.connectors.values():
        connector.connect = AsyncMock(return_value=True)
        connector.get_health_status = AsyncMock(return_value={"status": "healthy"})
    await orchestrator.initialize()
    
    # Mock the analysis methods
    orchestrator._analyze_error_impact = AsyncMock(return_value=[
        orchestrator.environments["prod-onprem"]
    ])
    orchestrator._are_environments_connected = AsyncMock(return_value=True)
    
    error_data = {
        "error_id": "err-123",
        "environment_id": "prod-aws",
        "type": "service_unavailable",
        "severity": 2,
        "service_dependencies": ["auth-service", "data-service"]
    }
    
    context = await orchestrator.detect_cross_environment_issue(error_data)
    
    assert context is not None
    assert context.error_id == "err-123"
    assert context.source_environment.id == "prod-aws"
    assert len(context.affected_environments) == 1
    assert context.scope == HealingScope.CROSS_ENVIRONMENT
    assert "auth-service" in context.dependencies


@pytest.mark.asyncio
async def test_create_healing_plan(orchestrator):
    """Test healing plan creation"""
    # Create mock context
    context = HealingContext(
        error_id="err-123",
        source_environment=orchestrator.environments["prod-aws"],
        affected_environments=[
            orchestrator.environments["prod-aws"],
            orchestrator.environments["prod-onprem"]
        ],
        scope=HealingScope.CROSS_ENVIRONMENT,
        dependencies=["service-a", "service-b"],
        constraints={"require_approval": True},
        priority=1,
        timestamp=datetime.utcnow()
    )
    
    patch_data = {
        "type": "code_fix",
        "file": "app.py",
        "changes": [{"line": 10, "content": "fixed_code()"}]
    }
    
    plan = await orchestrator.create_healing_plan(context, patch_data)
    
    assert plan is not None
    assert plan.context == context
    assert len(plan.steps) > 0
    assert plan.approval_required is True
    assert plan.risk_score > 0
    assert plan.plan_id in orchestrator.active_plans


@pytest.mark.asyncio
async def test_execute_healing_plan_success(orchestrator):
    """Test successful execution of healing plan"""
    # Create a simple plan
    env = orchestrator.environments["prod-aws"]
    context = HealingContext(
        error_id="err-123",
        source_environment=env,
        affected_environments=[env],
        scope=HealingScope.ENVIRONMENT,
        dependencies=[],
        constraints={},
        priority=5,
        timestamp=datetime.utcnow()
    )
    
    steps = [
        HealingStep(
            step_id="apply_prod-aws",
            environment=env,
            action="apply_patch",
            parameters={"patch": {"fix": "data"}},
            dependencies=[],
            timeout=60,
            can_fail=False,
            rollback_action="revert_patch"
        ),
        HealingStep(
            step_id="verify_prod-aws",
            environment=env,
            action="verify_healing",
            parameters={"error_id": "err-123"},
            dependencies=["apply_prod-aws"],
            timeout=30,
            can_fail=False,
            rollback_action=None
        )
    ]
    
    plan = HealingPlan(
        plan_id="test-plan-1",
        context=context,
        steps=steps,
        rollback_steps=[],
        approval_required=False,
        estimated_duration=90,
        risk_score=0.3
    )
    
    orchestrator.active_plans[plan.plan_id] = plan
    
    # Mock connector responses
    connector = orchestrator.connectors["prod-aws"]
    connector.execute_action = AsyncMock(return_value={"status": "success", "result": {}})
    
    # Mock auditor
    orchestrator.auditor.log_event = AsyncMock()
    
    result = await orchestrator.execute_healing_plan(plan)
    
    assert result["status"] == "success"
    assert result["plan_id"] == plan.plan_id
    assert len(result["results"]) == 2
    
    # Verify auditor was called
    assert orchestrator.auditor.log_event.call_count >= 2


@pytest.mark.asyncio
async def test_execute_healing_plan_with_rollback(orchestrator):
    """Test healing plan execution with rollback on failure"""
    env = orchestrator.environments["prod-aws"]
    context = HealingContext(
        error_id="err-456",
        source_environment=env,
        affected_environments=[env],
        scope=HealingScope.ENVIRONMENT,
        dependencies=[],
        constraints={},
        priority=5,
        timestamp=datetime.utcnow()
    )
    
    steps = [
        HealingStep(
            step_id="apply_prod-aws",
            environment=env,
            action="apply_patch",
            parameters={"patch": {"fix": "data"}},
            dependencies=[],
            timeout=60,
            can_fail=False,
            rollback_action="revert_patch"
        )
    ]
    
    rollback_steps = [
        HealingStep(
            step_id="rollback_apply_prod-aws",
            environment=env,
            action="revert_patch",
            parameters={"patch": {"fix": "data"}},
            dependencies=[],
            timeout=30,
            can_fail=False,
            rollback_action=None
        )
    ]
    
    plan = HealingPlan(
        plan_id="test-plan-2",
        context=context,
        steps=steps,
        rollback_steps=rollback_steps,
        approval_required=False,
        estimated_duration=60,
        risk_score=0.3
    )
    
    orchestrator.active_plans[plan.plan_id] = plan
    
    # Mock connector to fail on apply, succeed on rollback
    connector = orchestrator.connectors["prod-aws"]
    connector.execute_action = AsyncMock(side_effect=[
        {"status": "failed", "error": "patch failed"},
        {"status": "success", "result": {}}
    ])
    
    orchestrator.auditor.log_event = AsyncMock()
    
    result = await orchestrator.execute_healing_plan(plan)
    
    assert result["status"] == "failed"
    assert "rollback_results" in result
    assert len(result["rollback_results"]) == 1
    assert result["rollback_results"]["rollback_apply_prod-aws"]["status"] == "success"


@pytest.mark.asyncio
async def test_get_cross_environment_view(orchestrator):
    """Test cross-environment view generation"""
    # Initialize environments
    for connector in orchestrator.connectors.values():
        connector.connect = AsyncMock(return_value=True)
        connector.get_health_status = AsyncMock(return_value={
            "status": "healthy",
            "services": {},
            "metrics": {}
        })
    await orchestrator.initialize()
    
    # Add an active healing plan
    env = orchestrator.environments["prod-aws"]
    context = HealingContext(
        error_id="err-789",
        source_environment=env,
        affected_environments=[env],
        scope=HealingScope.ENVIRONMENT,
        dependencies=[],
        constraints={},
        priority=3,
        timestamp=datetime.utcnow()
    )
    
    plan = HealingPlan(
        plan_id="test-plan-3",
        context=context,
        steps=[],
        rollback_steps=[],
        approval_required=False,
        estimated_duration=60,
        risk_score=0.4
    )
    
    orchestrator.active_plans[plan.plan_id] = plan
    
    view = await orchestrator.get_cross_environment_view()
    
    assert len(view["environments"]) == 3
    assert view["health_summary"]["healthy"] == 3
    assert len(view["active_healings"]) == 1
    assert view["active_healings"][0]["plan_id"] == "test-plan-3"


@pytest.mark.asyncio
async def test_cloud_connector():
    """Test cloud connector functionality"""
    connector = CloudConnector("aws")
    
    # Test connection
    result = await connector.connect({"access_key": "test", "secret_key": "test"})
    assert result is True
    
    # Test health status
    health = await connector.get_health_status()
    assert health["status"] == "healthy"
    
    # Test execute action
    result = await connector.execute_action("apply_patch", {"patch": "data"})
    assert result["status"] == "success"
    
    # Test rollback
    result = await connector.rollback_action("apply_patch", {"patch": "data"})
    assert result is True


@pytest.mark.asyncio
async def test_onpremise_connector():
    """Test on-premise connector functionality"""
    connector = OnPremiseConnector()
    
    # Test connection
    result = await connector.connect({"host": "192.168.1.1", "port": 22})
    assert result is True
    
    # Test health status
    health = await connector.get_health_status()
    assert health["status"] == "healthy"
    
    # Test execute action
    result = await connector.execute_action("apply_patch", {"patch": "data"})
    assert result["status"] == "success"
    
    # Test rollback
    result = await connector.rollback_action("apply_patch", {"patch": "data"})
    assert result is True


def test_environment_ordering(orchestrator):
    """Test environment dependency ordering"""
    envs = list(orchestrator.environments.values())
    ordered = orchestrator._order_environments_by_dependency(envs)
    
    # Verify ordering by dependency level
    assert ordered[0].id == "prod-onprem"  # level 0
    assert ordered[1].id == "prod-aws"     # level 1
    assert ordered[2].id == "dev-gcp"      # level 2


def test_risk_score_calculation(orchestrator):
    """Test risk score calculation"""
    context = HealingContext(
        error_id="err-test",
        source_environment=orchestrator.environments["prod-aws"],
        affected_environments=[
            orchestrator.environments["prod-aws"],
            orchestrator.environments["prod-onprem"]
        ],
        scope=HealingScope.CROSS_ENVIRONMENT,
        dependencies=[],
        constraints={},
        priority=1,
        timestamp=datetime.utcnow()
    )
    
    steps = [HealingStep("s1", orchestrator.environments["prod-aws"], "action", {}, [], 60, False, None)] * 5
    
    risk_score = orchestrator._calculate_risk_score(context, steps)
    
    # Base risk (0.3) + cross-env (0.2) + 2 prod envs (0.2) + 5 steps (0.1)
    assert risk_score == pytest.approx(0.8, rel=0.01)


def test_duration_estimation(orchestrator):
    """Test healing duration estimation"""
    env = orchestrator.environments["prod-aws"]
    
    # Parallel steps
    steps = [
        HealingStep("s1", env, "action1", {}, [], 100, False, None),
        HealingStep("s2", env, "action2", {}, [], 150, False, None),
        HealingStep("s3", env, "action3", {}, ["s1", "s2"], 50, False, None)
    ]
    
    duration = orchestrator._estimate_duration(steps)
    # Max of parallel (150) + sequential (50) = 200
    assert duration == 200