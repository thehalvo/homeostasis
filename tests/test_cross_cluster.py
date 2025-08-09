"""
Tests for Cross-Cluster Orchestration
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from modules.deployment.multi_environment.cross_cluster import (
    CrossClusterOrchestrator,
    Cluster,
    ClusterType,
    ClusterState,
    ServiceMeshType,
    Service,
    CrossClusterPolicy,
    ClusterHealth,
    KubernetesConnector,
    ServiceMeshController
)
from modules.deployment.multi_environment.hybrid_orchestrator import (
    Environment, EnvironmentType, HealingContext, HealingScope, HealingPlan, HealingStep
)


@pytest.fixture
def mock_clusters():
    """Create mock clusters for testing"""
    return [
        {
            "id": "k8s-prod-1",
            "name": "Production Kubernetes 1",
            "type": "kubernetes",
            "endpoint": "https://k8s-prod-1.example.com",
            "auth": {
                "kubeconfig": "/path/to/kubeconfig",
                "context": "prod-1"
            },
            "region": "us-east-1",
            "service_mesh": "istio",
            "namespaces": ["default", "production"],
            "metadata": {"tier": "primary"}
        },
        {
            "id": "k8s-prod-2",
            "name": "Production Kubernetes 2",
            "type": "kubernetes",
            "endpoint": "https://k8s-prod-2.example.com",
            "auth": {
                "kubeconfig": "/path/to/kubeconfig",
                "context": "prod-2"
            },
            "region": "us-west-2",
            "service_mesh": "istio",
            "namespaces": ["default", "production"],
            "metadata": {"tier": "secondary"}
        },
        {
            "id": "eks-prod",
            "name": "Production EKS",
            "type": "eks",
            "endpoint": "https://eks-prod.amazonaws.com",
            "auth": {
                "kubeconfig": "/path/to/eks-kubeconfig"
            },
            "region": "eu-west-1",
            "service_mesh": "linkerd",
            "namespaces": ["default", "production"],
            "metadata": {"tier": "primary"}
        }
    ]


@pytest.fixture
def mock_services():
    """Create mock services for testing"""
    return [
        {
            "name": "api-service",
            "namespace": "production",
            "version": "v1.2.3",
            "replicas": {"k8s-prod-1": 3, "k8s-prod-2": 2},
            "endpoints": {
                "k8s-prod-1": "api-service.prod-1.local",
                "k8s-prod-2": "api-service.prod-2.local"
            },
            "health_checks": ["/health", "/ready"],
            "dependencies": ["auth-service", "database"],
            "metadata": {"tier": "critical"}
        },
        {
            "name": "auth-service",
            "namespace": "production",
            "version": "v2.0.1",
            "replicas": {"k8s-prod-1": 2, "eks-prod": 2},
            "endpoints": {
                "k8s-prod-1": "auth-service.prod-1.local",
                "eks-prod": "auth-service.eks.local"
            },
            "health_checks": ["/health"],
            "dependencies": ["database"],
            "metadata": {"tier": "critical"}
        }
    ]


@pytest.fixture
def mock_policies():
    """Create mock cross-cluster policies"""
    return [
        {
            "name": "balanced-distribution",
            "distribution_strategy": "balanced",
            "min_replicas_per_cluster": 1,
            "max_replicas_per_cluster": 10,
            "failover_enabled": True,
            "auto_scaling_enabled": True,
            "traffic_split": {
                "k8s-prod-1": 0.5,
                "k8s-prod-2": 0.3,
                "eks-prod": 0.2
            },
            "consistency_requirements": "eventual"
        },
        {
            "name": "primary-backup",
            "distribution_strategy": "primary-backup",
            "min_replicas_per_cluster": 1,
            "max_replicas_per_cluster": 20,
            "failover_enabled": True,
            "auto_scaling_enabled": False,
            "consistency_requirements": "strong"
        }
    ]


@pytest.fixture
def mock_config(mock_clusters, mock_services, mock_policies):
    """Create mock configuration"""
    return {
        "clusters": mock_clusters,
        "services": mock_services,
        "policies": mock_policies
    }


@pytest.fixture
def orchestrator(mock_config):
    """Create orchestrator instance for testing"""
    with patch('modules.deployment.multi_environment.cross_cluster.DistributedMonitor'):
        return CrossClusterOrchestrator(mock_config)


@pytest.mark.asyncio
async def test_initialization(orchestrator):
    """Test orchestrator initialization"""
    # Mock connector methods
    for connector in orchestrator.connectors.values():
        connector.connect = AsyncMock(return_value=True)
        connector.get_health = AsyncMock(return_value=ClusterHealth(
            cluster_id="test",
            timestamp=datetime.utcnow(),
            node_count=5,
            ready_nodes=5,
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            pod_count=100,
            service_count=20,
            error_rate=0.01,
            latency_p99_ms=100,
            issues=[]
        ))
    
    result = await orchestrator.initialize()
    assert result is True
    
    # Verify all clusters were initialized
    assert len(orchestrator.clusters) == 3
    assert all(c.state == ClusterState.ACTIVE for c in orchestrator.clusters.values())


@pytest.mark.asyncio
async def test_kubernetes_connector():
    """Test Kubernetes connector functionality"""
    connector = KubernetesConnector("test-cluster")
    
    # Mock KubernetesManager
    with patch('modules.deployment.multi_environment.cross_cluster.KubernetesManager') as mock_k8s:
        mock_manager = MagicMock()
        mock_k8s.return_value = mock_manager
        mock_manager.get_cluster_version = AsyncMock(return_value="1.25.0")
        
        # Test connection
        result = await connector.connect("https://test.k8s.io", {"kubeconfig": "/test"})
        assert result is True
        assert connector.k8s_manager is not None
        
        # Test get health
        mock_manager.list_nodes = AsyncMock(return_value=[
            {"name": "node1", "status": "Ready"},
            {"name": "node2", "status": "Ready"}
        ])
        mock_manager.get_cluster_metrics = AsyncMock(return_value={
            "cpu_usage": 45.0,
            "memory_usage": 55.0,
            "error_rate": 0.02,
            "latency_p99": 150
        })
        mock_manager.list_all_pods = AsyncMock(return_value=[{}, {}, {}])
        mock_manager.list_all_services = AsyncMock(return_value=[{}, {}])
        
        health = await connector.get_health()
        assert health.node_count == 2
        assert health.ready_nodes == 2
        assert health.cpu_usage_percent == 45.0
        assert health.pod_count == 3


@pytest.mark.asyncio
async def test_deploy_service_across_clusters(orchestrator):
    """Test deploying a service across multiple clusters"""
    # Mock connectors
    for cluster_id, connector in orchestrator.connectors.items():
        connector.deploy_service = AsyncMock(return_value=True)
    
    # Mock service mesh configuration
    for controller in orchestrator.mesh_controllers.values():
        controller.enable_circuit_breaking = AsyncMock(return_value=True)
        controller.configure_retry_policy = AsyncMock(return_value=True)
    
    result = await orchestrator.deploy_service_across_clusters(
        "api-service",
        target_clusters=["k8s-prod-1", "k8s-prod-2"],
        policy_name="balanced-distribution"
    )
    
    assert result["service"] == "api-service"
    assert len(result["deployments"]) == 2
    assert all(d["success"] for d in result["deployments"].values())
    assert result["total_replicas"] > 0


def test_replica_distribution_balanced(orchestrator):
    """Test balanced replica distribution calculation"""
    service = orchestrator.services["api-service"]
    policy = orchestrator.policies["balanced-distribution"]
    clusters = ["k8s-prod-1", "k8s-prod-2", "eks-prod"]
    
    distribution = orchestrator._calculate_replica_distribution(service, clusters, policy)
    
    # Should distribute evenly
    assert len(distribution) == 3
    assert all(1 <= replicas <= 10 for replicas in distribution.values())
    # Total should be close to original
    assert abs(sum(distribution.values()) - sum(service.replicas.values())) <= len(clusters)


def test_replica_distribution_primary_backup(orchestrator):
    """Test primary-backup replica distribution"""
    service = orchestrator.services["api-service"]
    policy = orchestrator.policies["primary-backup"]
    clusters = ["k8s-prod-1", "k8s-prod-2", "eks-prod"]
    
    distribution = orchestrator._calculate_replica_distribution(service, clusters, policy)
    
    # First cluster should have most replicas
    assert distribution["k8s-prod-1"] > distribution["k8s-prod-2"]
    assert distribution["k8s-prod-1"] > distribution["eks-prod"]
    # Backup clusters should have minimum
    assert distribution["k8s-prod-2"] == policy.min_replicas_per_cluster
    assert distribution["eks-prod"] == policy.min_replicas_per_cluster


@pytest.mark.asyncio
async def test_handle_cluster_failure(orchestrator):
    """Test handling of cluster failure"""
    # Set up initial state
    orchestrator.clusters["k8s-prod-1"].state = ClusterState.ACTIVE
    orchestrator.clusters["k8s-prod-2"].state = ClusterState.ACTIVE
    orchestrator.clusters["eks-prod"].state = ClusterState.ACTIVE
    
    # Mock connectors
    for connector in orchestrator.connectors.values():
        connector.scale_service = AsyncMock(return_value=True)
    
    # Mock traffic rebalancing
    orchestrator._rebalance_traffic = AsyncMock()
    
    # Simulate cluster failure
    result = await orchestrator.handle_cluster_failure("k8s-prod-1")
    
    assert orchestrator.clusters["k8s-prod-1"].state == ClusterState.FAILED
    assert "affected_services" in result
    assert "api-service" in result["affected_services"]
    assert "auth-service" in result["affected_services"]
    
    # Verify services were redistributed
    assert "redistribution" in result
    # Verify failed cluster was removed from services
    assert "k8s-prod-1" not in orchestrator.services["api-service"].replicas


@pytest.mark.asyncio
async def test_cross_cluster_healing(orchestrator):
    """Test cross-cluster healing coordination"""
    env = Environment("env1", "Env 1", EnvironmentType.CLOUD_AWS, "us-east-1", {}, [], "healthy", {})
    
    healing_context = HealingContext(
        error_id="err-123",
        source_environment=env,
        affected_environments=[env],
        scope=HealingScope.CROSS_ENVIRONMENT,
        dependencies=["api-service"],
        constraints={"patch": {"spec": {"template": {"metadata": {"labels": {"fix": "applied"}}}}}},
        priority=1,
        timestamp=datetime.utcnow()
    )
    
    healing_plan = HealingPlan(
        plan_id="plan-123",
        context=healing_context,
        steps=[
            HealingStep("step1", env, "patch", {"service": "api-service"}, [], 60, False, None)
        ],
        rollback_steps=[],
        approval_required=False,
        estimated_duration=60,
        risk_score=0.3
    )
    
    # Mock connectors
    for connector in orchestrator.connectors.values():
        connector.apply_patch = AsyncMock(return_value=True)
    
    result = await orchestrator.perform_cross_cluster_healing(healing_context, healing_plan)
    
    assert result["services"]["api-service"]["k8s-prod-1"]["patch_applied"] is True
    assert "k8s-prod-1" in result["clusters"]


@pytest.mark.asyncio
async def test_service_mesh_controller_istio():
    """Test Istio service mesh configuration"""
    controller = ServiceMeshController(ServiceMeshType.ISTIO)
    
    service = Service(
        name="test-service",
        namespace="default",
        version="v1",
        replicas={},
        endpoints={},
        health_checks=[],
        dependencies=[],
        metadata={}
    )
    
    # Test traffic split configuration
    result = await controller.configure_traffic_split(
        service,
        {"cluster-1": 0.6, "cluster-2": 0.4}
    )
    assert result is True
    
    # Test circuit breaking
    result = await controller.enable_circuit_breaking(
        service,
        {"consecutive_errors": 5, "interval_seconds": 30}
    )
    assert result is True


@pytest.mark.asyncio
async def test_get_cluster_status(orchestrator):
    """Test getting cluster status"""
    cluster_id = "k8s-prod-1"
    
    # Mock connector
    connector = orchestrator.connectors[cluster_id]
    connector.get_health = AsyncMock(return_value=ClusterHealth(
        cluster_id=cluster_id,
        timestamp=datetime.utcnow(),
        node_count=5,
        ready_nodes=5,
        cpu_usage_percent=45.0,
        memory_usage_percent=55.0,
        pod_count=50,
        service_count=10,
        error_rate=0.01,
        latency_p99_ms=100,
        issues=[]
    ))
    
    connector.get_service_status = AsyncMock(return_value={
        "exists": True,
        "ready": 3,
        "desired": 3,
        "pods": 3,
        "version": "v1.2.3"
    })
    
    status = await orchestrator.get_cluster_status(cluster_id)
    
    assert status["cluster"]["id"] == cluster_id
    assert status["cluster"]["state"] == "active"
    assert status["health"]["nodes"] == "5/5"
    assert status["health"]["cpu_usage"] == 45.0
    assert "api-service" in status["services"]


@pytest.mark.asyncio
async def test_get_cross_cluster_view(orchestrator):
    """Test getting cross-cluster view"""
    # Mock get_cluster_status
    orchestrator.get_cluster_status = AsyncMock(return_value={
        "cluster": {"id": "test", "state": "active"},
        "health": {"nodes": "5/5"}
    })
    
    view = await orchestrator.get_cross_cluster_view()
    
    assert len(view["clusters"]) == 3
    assert len(view["services"]) == 2
    assert len(view["policies"]) == 2
    assert view["health_summary"]["total_clusters"] == 3
    assert view["health_summary"]["total_services"] == 2


@pytest.mark.asyncio
async def test_rebalance_clusters(orchestrator):
    """Test cluster rebalancing"""
    # Set all clusters active
    for cluster in orchestrator.clusters.values():
        cluster.state = ClusterState.ACTIVE
    
    # Mock connectors
    for connector in orchestrator.connectors.values():
        connector.scale_service = AsyncMock(return_value=True)
    
    # Mock traffic rebalancing
    orchestrator._rebalance_traffic = AsyncMock()
    
    result = await orchestrator.rebalance_clusters("balanced-distribution")
    
    assert result["policy"] == "balanced-distribution"
    assert len(result["active_clusters"]) == 3
    
    # Verify rebalancing was attempted
    orchestrator._rebalance_traffic.assert_called()


def test_deployment_config_generation(orchestrator):
    """Test deployment configuration generation"""
    service = orchestrator.services["api-service"]
    config = orchestrator._get_deployment_config(service, "k8s-prod-1")
    
    assert config["image"] == "api-service:v1.2.3"
    assert any(env["name"] == "CLUSTER_ID" for env in config["env"])
    assert any(env["name"] == "REGION" for env in config["env"])
    assert config["expose"] is True
    assert "resources" in config


@pytest.mark.asyncio
async def test_traffic_rebalancing(orchestrator):
    """Test traffic rebalancing after cluster changes"""
    service = orchestrator.services["api-service"]
    active_clusters = ["k8s-prod-1", "k8s-prod-2"]
    
    # Mock traffic split update
    orchestrator._update_traffic_split = AsyncMock()
    
    await orchestrator._rebalance_traffic(service, active_clusters)
    
    # Verify traffic split was calculated correctly
    orchestrator._update_traffic_split.assert_called_once()
    call_args = orchestrator._update_traffic_split.call_args[0]
    traffic_split = call_args[1]
    
    # Should split based on replica count
    total_replicas = sum(service.replicas.get(c, 0) for c in active_clusters)
    expected_split = {
        c: service.replicas.get(c, 0) / total_replicas
        for c in active_clusters
    }
    
    assert traffic_split == expected_split


@pytest.mark.asyncio
async def test_cluster_failure_no_services(orchestrator):
    """Test cluster failure with no affected services"""
    # Create a cluster with no services
    orchestrator.services = {}
    
    result = await orchestrator.handle_cluster_failure("k8s-prod-1")
    
    assert result["message"] == "No services affected"


@pytest.mark.asyncio
async def test_cluster_failure_no_healthy_clusters(orchestrator):
    """Test cluster failure when no healthy clusters available"""
    # Set all clusters to failed state
    for cluster in orchestrator.clusters.values():
        cluster.state = ClusterState.FAILED
    
    result = await orchestrator.handle_cluster_failure("k8s-prod-1")
    
    assert "error" in result
    assert "No healthy clusters available" in result["error"]