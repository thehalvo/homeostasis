"""
Tests for Multi-Region Resilience Strategies
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from modules.deployment.multi_environment.multi_region import (
    MultiRegionResilienceStrategy,
    Region,
    RegionStatus,
    RegionHealth,
    FailoverStrategy,
    ConsistencyModel,
    ResiliencePolicy,
    RegionHealthMonitor,
    FailoverOrchestrator,
    FailoverEvent
)
from modules.deployment.multi_environment.hybrid_orchestrator import (
    Environment, EnvironmentType, HealingContext, HealingScope, HealingPlan, HealingStep
)


@pytest.fixture
def mock_regions():
    """Create mock regions for testing"""
    return [
        Region(
            id="us-east-1",
            name="US East 1",
            location=(40.7128, -74.0060),  # New York
            environments=[],
            status=RegionStatus.HEALTHY,
            capacity={"cpu": 1000, "memory": 2000, "disk": 5000},
            metadata={"tier": "primary"}
        ),
        Region(
            id="us-west-2",
            name="US West 2",
            location=(47.6062, -122.3321),  # Seattle
            environments=[],
            status=RegionStatus.HEALTHY,
            capacity={"cpu": 800, "memory": 1600, "disk": 4000},
            metadata={"tier": "secondary"}
        ),
        Region(
            id="eu-west-1",
            name="EU West 1",
            location=(53.3498, -6.2603),  # Dublin
            environments=[],
            status=RegionStatus.HEALTHY,
            capacity={"cpu": 900, "memory": 1800, "disk": 4500},
            metadata={"tier": "primary"}
        ),
        Region(
            id="ap-south-1",
            name="AP South 1",
            location=(19.0760, 72.8777),  # Mumbai
            environments=[],
            status=RegionStatus.DEGRADED,
            capacity={"cpu": 700, "memory": 1400, "disk": 3500},
            metadata={"tier": "secondary"}
        )
    ]


@pytest.fixture
def mock_policies():
    """Create mock resilience policies"""
    return [
        {
            "name": "primary_policy",
            "failover_strategy": "active_passive",
            "consistency_model": "strong",
            "rpo_seconds": 30,
            "rto_seconds": 300,
            "health_check_interval": 30,
            "failover_threshold": 0.8,
            "auto_failback": True,
            "geo_restrictions": ["us-east-1", "us-west-2", "eu-west-1"]
        },
        {
            "name": "global_policy",
            "failover_strategy": "active_active",
            "consistency_model": "eventual",
            "rpo_seconds": 60,
            "rto_seconds": 600,
            "health_check_interval": 60,
            "failover_threshold": 0.6,
            "auto_failback": False
        }
    ]


@pytest.fixture
def mock_config(mock_regions, mock_policies):
    """Create mock configuration"""
    return {
        "regions": [
            {
                "id": r.id,
                "name": r.name,
                "location": list(r.location),
                "capacity": r.capacity,
                "metadata": r.metadata
            }
            for r in mock_regions
        ],
        "policies": mock_policies
    }


@pytest.fixture
def strategy(mock_config):
    """Create strategy instance for testing"""
    with patch('modules.deployment.multi_environment.multi_region.DistributedMonitor'):
        with patch('modules.deployment.multi_environment.multi_region.SecurityAuditor'):
            return MultiRegionResilienceStrategy(mock_config)


@pytest.mark.asyncio
async def test_region_health_monitor(mock_regions):
    """Test region health monitoring"""
    monitor = RegionHealthMonitor(mock_regions)
    
    # Mock environment health check
    monitor._check_environment_health = AsyncMock(return_value={
        "availability": 95.0,
        "latency_ms": 50.0,
        "error_rate": 0.02,
        "cpu_usage": 60.0,
        "memory_usage": 70.0,
        "disk_usage": 40.0
    })
    
    # Check health of a region
    health = await monitor.check_region_health(mock_regions[0])
    
    assert health.region_id == "us-east-1"
    assert health.status == RegionStatus.HEALTHY
    assert health.availability == 95.0
    assert health.latency_ms == 50.0
    assert health.error_rate == 0.02
    assert "cpu" in health.capacity_usage
    
    # Verify health history is stored
    assert len(monitor.health_history["us-east-1"]) == 1


@pytest.mark.asyncio
async def test_region_health_degraded(mock_regions):
    """Test detection of degraded region health"""
    monitor = RegionHealthMonitor(mock_regions)
    
    # Mock poor health metrics
    monitor._check_environment_health = AsyncMock(return_value={
        "availability": 85.0,
        "latency_ms": 200.0,
        "error_rate": 0.07,
        "cpu_usage": 90.0,
        "memory_usage": 95.0,
        "disk_usage": 80.0
    })
    
    health = await monitor.check_region_health(mock_regions[0])
    
    assert health.status == RegionStatus.DEGRADED
    assert health.availability == 85.0
    assert health.error_rate == 0.07


def test_region_trend_analysis(mock_regions):
    """Test region health trend analysis"""
    monitor = RegionHealthMonitor(mock_regions)
    
    # Add historical health data
    now = datetime.utcnow()
    for i in range(10):
        health = RegionHealth(
            region_id="us-east-1",
            status=RegionStatus.HEALTHY if i < 5 else RegionStatus.DEGRADED,
            availability=95.0 if i < 5 else 85.0,
            latency_ms=50.0 + i * 10,
            error_rate=0.01 + i * 0.01,
            capacity_usage={"cpu": 50 + i * 5},
            last_updated=now - timedelta(hours=10-i),
            incidents=[]
        )
        monitor.health_history["us-east-1"].append(health)
    
    # Get trend analysis
    trend = monitor.get_region_trend("us-east-1", hours=24)
    
    assert trend["average_availability"] == 90.0
    assert trend["trend"] == "degrading"
    assert trend["status_counts"]["healthy"] == 5
    assert trend["status_counts"]["degraded"] == 5


@pytest.mark.asyncio
async def test_failover_orchestrator_active_passive():
    """Test active-passive failover execution"""
    regions = [
        Region("region1", "Region 1", (0, 0), [], RegionStatus.UNHEALTHY, {}, {}),
        Region("region2", "Region 2", (10, 10), [], RegionStatus.HEALTHY, {}, {})
    ]
    
    policy = ResiliencePolicy(
        name="test_policy",
        failover_strategy=FailoverStrategy.ACTIVE_PASSIVE,
        consistency_model=ConsistencyModel.STRONG,
        rpo_seconds=30,
        rto_seconds=300,
        health_check_interval=30,
        failover_threshold=0.8,
        auto_failback=True,
        geo_restrictions=None
    )
    
    with patch('modules.deployment.multi_environment.multi_region.SecurityAuditor'):
        orchestrator = FailoverOrchestrator(regions, policy)
        
        # Mock methods
        orchestrator._activate_region = AsyncMock()
        orchestrator._deactivate_region = AsyncMock()
        orchestrator._update_routing = AsyncMock()
        orchestrator._verify_failover = AsyncMock(return_value=True)
        
        # Execute failover
        event = await orchestrator.execute_failover(
            "region1", "region2", "health degradation", ["service1", "service2"]
        )
        
        assert event.success is True
        assert event.from_region.id == "region1"
        assert event.to_region.id == "region2"
        assert event.duration_ms is not None
        
        # Verify methods were called
        orchestrator._activate_region.assert_called_with("region2")
        orchestrator._deactivate_region.assert_called_with("region1")


@pytest.mark.asyncio
async def test_failover_pre_checks():
    """Test failover pre-flight checks"""
    regions = [
        Region("region1", "Region 1", (0, 0), [], RegionStatus.HEALTHY, {}, {}),
        Region("region2", "Region 2", (10, 10), [], RegionStatus.UNHEALTHY, {}, {})
    ]
    
    policy = ResiliencePolicy(
        name="test_policy",
        failover_strategy=FailoverStrategy.ACTIVE_PASSIVE,
        consistency_model=ConsistencyModel.STRONG,
        rpo_seconds=30,
        rto_seconds=300,
        health_check_interval=30,
        failover_threshold=0.8,
        auto_failback=True,
        geo_restrictions=["region1"]
    )
    
    with patch('modules.deployment.multi_environment.multi_region.SecurityAuditor'):
        orchestrator = FailoverOrchestrator(regions, policy)
        
        # Test failover to unhealthy region
        with pytest.raises(Exception, match="not healthy"):
            await orchestrator._pre_failover_checks("region1", "region2")
        
        # Test geo-restriction violation
        regions[1].status = RegionStatus.HEALTHY
        with pytest.raises(Exception, match="not allowed by policy"):
            await orchestrator._pre_failover_checks("region1", "region2")


@pytest.mark.asyncio
async def test_multi_region_healing_coordination(strategy):
    """Test healing coordination across multiple regions"""
    # Create mock healing context
    env1 = Environment("env1", "Env 1", EnvironmentType.CLOUD_AWS, "us-east-1", {}, [], "healthy", {})
    env2 = Environment("env2", "Env 2", EnvironmentType.CLOUD_AWS, "us-west-2", {}, [], "healthy", {})
    
    # Add environments to regions
    strategy.regions[0].environments.append(env1)
    strategy.regions[1].environments.append(env2)
    
    context = HealingContext(
        error_id="err-123",
        source_environment=env1,
        affected_environments=[env1, env2],
        scope=HealingScope.CROSS_ENVIRONMENT,
        dependencies=["payment", "auth"],
        constraints={"data_consistency": True},
        priority=1,
        timestamp=datetime.utcnow()
    )
    
    plan = HealingPlan(
        plan_id="plan-123",
        context=context,
        steps=[
            HealingStep("step1", env1, "action1", {}, [], 60, False, None),
            HealingStep("step2", env2, "action2", {}, ["step1"], 60, False, None)
        ],
        rollback_steps=[],
        approval_required=False,
        estimated_duration=120,
        risk_score=0.5
    )
    
    # Mock coordination methods
    strategy._coordinate_synchronous_healing = AsyncMock(return_value={
        "status": "synchronous",
        "regions": {"us-east-1": {"status": "success"}, "us-west-2": {"status": "success"}}
    })
    
    result = await strategy.handle_healing_across_regions(context, plan)
    
    assert result["status"] == "synchronous"
    assert len(result["regions"]) == 2
    strategy._coordinate_synchronous_healing.assert_called_once()


def test_consistency_model_determination(strategy):
    """Test determination of consistency model based on context"""
    env = Environment("env1", "Env 1", EnvironmentType.CLOUD_AWS, "us-east-1", {}, [], "healthy", {})
    
    # Test with data consistency requirement
    context = HealingContext(
        error_id="err-1",
        source_environment=env,
        affected_environments=[env],
        scope=HealingScope.ENVIRONMENT,
        dependencies=["service1"],
        constraints={"data_consistency": True},
        priority=1,
        timestamp=datetime.utcnow()
    )
    
    model = strategy._determine_consistency_model(context)
    assert model == ConsistencyModel.STRONG
    
    # Test with critical service
    context.constraints = {"data_consistency": False}
    context.dependencies = ["payment", "service2"]
    
    model = strategy._determine_consistency_model(context)
    assert model == ConsistencyModel.STRONG
    
    # Test with non-critical service
    context.dependencies = ["logging", "metrics"]
    
    model = strategy._determine_consistency_model(context)
    assert model == ConsistencyModel.EVENTUAL


def test_step_phase_grouping(strategy):
    """Test grouping of healing steps into phases"""
    env = Environment("env1", "Env 1", EnvironmentType.CLOUD_AWS, "us-east-1", {}, [], "healthy", {})
    
    steps = [
        HealingStep("step1", env, "action1", {}, [], 60, False, None),
        HealingStep("step2", env, "action2", {}, [], 60, False, None),
        HealingStep("step3", env, "action3", {}, ["step1", "step2"], 60, False, None),
        HealingStep("step4", env, "action4", {}, ["step3"], 60, False, None)
    ]
    
    phases = strategy._group_steps_by_phase(steps)
    
    assert len(phases) == 3
    assert len(phases["phase_0"]) == 2  # step1 and step2 (no dependencies)
    assert len(phases["phase_1"]) == 1  # step3 (depends on step1 and step2)
    assert len(phases["phase_2"]) == 1  # step4 (depends on step3)


def test_region_distance_calculation(strategy):
    """Test distance calculation between regions"""
    # Distance between New York and Seattle
    distance = strategy.get_region_distance("us-east-1", "us-west-2")
    assert 3800 < distance < 4000  # Approximately 3,900 km
    
    # Distance between New York and Dublin
    distance = strategy.get_region_distance("us-east-1", "eu-west-1")
    assert 5000 < distance < 5500  # Approximately 5,100 km
    
    # Invalid region
    distance = strategy.get_region_distance("us-east-1", "invalid")
    assert distance == float('inf')


def test_optimal_region_selection(strategy):
    """Test selection of optimal region for a location"""
    # Location near Seattle
    seattle_location = (47.6, -122.3)
    optimal = strategy.get_optimal_region_for_location(seattle_location)
    assert optimal == "us-west-2"
    
    # Location near Dublin
    dublin_location = (53.3, -6.2)
    optimal = strategy.get_optimal_region_for_location(dublin_location)
    assert optimal == "eu-west-1"
    
    # With restricted regions
    optimal = strategy.get_optimal_region_for_location(
        seattle_location,
        available_regions=["us-east-1", "eu-west-1"]
    )
    assert optimal == "us-east-1"  # Closest available


@pytest.mark.asyncio
async def test_resilience_status_generation(strategy):
    """Test comprehensive resilience status generation"""
    # Mock health monitor
    strategy.health_monitor.check_region_health = AsyncMock(return_value=RegionHealth(
        region_id="us-east-1",
        status=RegionStatus.HEALTHY,
        availability=95.0,
        latency_ms=50.0,
        error_rate=0.02,
        capacity_usage={"cpu": 60, "memory": 70, "disk": 40},
        last_updated=datetime.utcnow(),
        incidents=[]
    ))
    
    # Add failover history
    event = FailoverEvent(
        event_id="failover-1",
        timestamp=datetime.utcnow(),
        from_region=strategy.regions[0],
        to_region=strategy.regions[1],
        reason="test",
        affected_services=["service1"],
        duration_ms=5000,
        success=True
    )
    strategy.failover_orchestrators["primary_policy"].failover_history.append(event)
    
    status = await strategy.get_resilience_status()
    
    assert len(status["regions"]) == 4
    assert "us-east-1" in status["regions"]
    assert status["regions"]["us-east-1"]["availability"] == 95.0
    
    assert len(status["policies"]) == 2
    assert "primary_policy" in status["policies"]
    
    assert len(status["failover_history"]) == 1
    assert status["failover_history"][0]["success"] is True
    
    assert isinstance(status["recommendations"], list)


def test_recommendation_generation(strategy):
    """Test generation of recommendations based on status"""
    status = {
        "regions": {
            "us-east-1": {
                "name": "US East 1",
                "status": "degraded",
                "availability": 75.0,
                "capacity_usage": {"cpu": 85, "memory": 70, "disk": 40}
            },
            "us-west-2": {
                "name": "US West 2",
                "status": "healthy",
                "availability": 95.0,
                "capacity_usage": {"cpu": 60, "memory": 50, "disk": 30}
            }
        },
        "failover_history": [
            {"success": False},
            {"success": False},
            {"success": False}
        ],
        "health_trends": {
            "us-east-1": {"trend": "degrading"}
        }
    }
    
    recommendations = strategy._generate_recommendations(status)
    
    # Should recommend failover for degraded region
    assert any("Consider failover" in r for r in recommendations)
    
    # Should flag high CPU usage
    assert any("High cpu usage" in r for r in recommendations)
    
    # Should flag multiple failover failures
    assert any("Multiple failover failures" in r for r in recommendations)
    
    # Should flag degrading trend
    assert any("degrading health trend" in r for r in recommendations)


@pytest.mark.asyncio
async def test_auto_failback():
    """Test automatic failback functionality"""
    regions = [
        Region("region1", "Region 1", (0, 0), [], RegionStatus.HEALTHY, {}, {}),
        Region("region2", "Region 2", (10, 10), [], RegionStatus.HEALTHY, {}, {})
    ]
    
    policy = ResiliencePolicy(
        name="test_policy",
        failover_strategy=FailoverStrategy.PRIMARY_SECONDARY,
        consistency_model=ConsistencyModel.STRONG,
        rpo_seconds=30,
        rto_seconds=300,
        health_check_interval=30,
        failover_threshold=0.8,
        auto_failback=True,
        geo_restrictions=None
    )
    
    with patch('modules.deployment.multi_environment.multi_region.SecurityAuditor'):
        orchestrator = FailoverOrchestrator(regions, policy)
        orchestrator.primary_region = "region2"
        orchestrator.active_regions = {"region2"}
        
        # Mock execute_failover
        orchestrator.execute_failover = AsyncMock(return_value=FailoverEvent(
            event_id="failback-1",
            timestamp=datetime.utcnow(),
            from_region=regions[1],
            to_region=regions[0],
            reason="auto_failback",
            affected_services=[],
            duration_ms=3000,
            success=True
        ))
        
        # Execute auto failback
        event = await orchestrator.auto_failback("region1")
        
        assert event is not None
        assert event.reason == "auto_failback"
        orchestrator.execute_failover.assert_called_once()


@pytest.mark.asyncio
async def test_health_monitoring_loop(strategy):
    """Test continuous health monitoring loop"""
    # Mock methods
    strategy.health_monitor.check_region_health = AsyncMock(return_value=RegionHealth(
        region_id="us-east-1",
        status=RegionStatus.HEALTHY,
        availability=95.0,
        latency_ms=50.0,
        error_rate=0.02,
        capacity_usage={},
        last_updated=datetime.utcnow(),
        incidents=[]
    ))
    
    strategy._check_failover_conditions = AsyncMock()
    
    # Run monitoring loop for a short time
    monitoring_task = asyncio.create_task(strategy.monitor_health_loop())
    await asyncio.sleep(0.1)
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    # Verify health checks were performed
    assert strategy.health_monitor.check_region_health.called
    assert strategy._check_failover_conditions.called