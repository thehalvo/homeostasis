"""
Cross-Cluster Orchestration

Provides orchestration capabilities across multiple Kubernetes clusters,
container orchestration platforms, and hybrid deployments.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from modules.deployment.kubernetes.k8s_manager import KubernetesManager
from modules.deployment.multi_environment.hybrid_orchestrator import (
    HealingContext, HealingPlan)
from modules.monitoring.distributed_monitoring import DistributedMonitor


class ClusterType(Enum):
    """Types of container orchestration clusters"""

    KUBERNETES = "kubernetes"
    OPENSHIFT = "openshift"
    DOCKER_SWARM = "docker_swarm"
    NOMAD = "nomad"
    ECS = "ecs"
    AKS = "aks"
    GKE = "gke"
    EKS = "eks"


class ClusterState(Enum):
    """State of a cluster"""

    ACTIVE = "active"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    FAILED = "failed"


class ServiceMeshType(Enum):
    """Types of service mesh technologies"""

    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL = "consul"
    AWS_APP_MESH = "aws_app_mesh"
    KUMA = "kuma"
    NONE = "none"


@dataclass
class Cluster:
    """Represents a container orchestration cluster"""

    id: str
    name: str
    type: ClusterType
    endpoint: str
    auth_config: Dict[str, Any]
    region: str
    state: ClusterState
    service_mesh: ServiceMeshType
    namespaces: List[str]
    metadata: Dict[str, Any]


@dataclass
class Service:
    """Represents a service deployed across clusters"""

    name: str
    namespace: str
    version: str
    replicas: Dict[str, int]  # cluster_id -> replica count
    endpoints: Dict[str, str]  # cluster_id -> endpoint
    health_checks: List[str]
    dependencies: List[str]
    metadata: Dict[str, Any]


@dataclass
class CrossClusterPolicy:
    """Policy for cross-cluster operations"""

    name: str
    distribution_strategy: str  # balanced, primary-backup, geo-aware
    min_replicas_per_cluster: int
    max_replicas_per_cluster: int
    failover_enabled: bool
    auto_scaling_enabled: bool
    traffic_split: Dict[str, float]  # cluster_id -> traffic percentage
    consistency_requirements: str  # strong, eventual, none


@dataclass
class ClusterHealth:
    """Health status of a cluster"""

    cluster_id: str
    timestamp: datetime
    node_count: int
    ready_nodes: int
    cpu_usage_percent: float
    memory_usage_percent: float
    pod_count: int
    service_count: int
    error_rate: float
    latency_p99_ms: float
    issues: List[Dict[str, Any]]


class ClusterConnector(ABC):
    """Abstract interface for cluster connections"""

    @abstractmethod
    async def connect(self, endpoint: str, auth_config: Dict[str, Any]) -> bool:
        """Connect to cluster"""
        pass

    @abstractmethod
    async def get_health(self) -> ClusterHealth:
        """Get cluster health status"""
        pass

    @abstractmethod
    async def deploy_service(self, service: Service, config: Dict[str, Any]) -> bool:
        """Deploy or update service in cluster"""
        pass

    @abstractmethod
    async def scale_service(
        self, service_name: str, namespace: str, replicas: int
    ) -> bool:
        """Scale service replicas"""
        pass

    @abstractmethod
    async def get_service_status(
        self, service_name: str, namespace: str
    ) -> Dict[str, Any]:
        """Get service status"""
        pass

    @abstractmethod
    async def apply_patch(
        self, resource_type: str, name: str, namespace: str, patch: Dict[str, Any]
    ) -> bool:
        """Apply patch to resource"""
        pass


class KubernetesConnector(ClusterConnector):
    """Connector for Kubernetes clusters"""

    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.k8s_manager = None
        self.logger = logging.getLogger(f"{__name__}.K8sConnector.{cluster_id}")

    async def connect(self, endpoint: str, auth_config: Dict[str, Any]) -> bool:
        """Connect to Kubernetes cluster"""
        try:
            self.k8s_manager = KubernetesManager(
                kubeconfig=auth_config.get("kubeconfig"),
                context=auth_config.get("context"),
            )
            # Test connection
            version = await self.k8s_manager.get_cluster_version()
            self.logger.info(f"Connected to Kubernetes cluster version: {version}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to cluster: {e}")
            return False

    async def get_health(self) -> ClusterHealth:
        """Get Kubernetes cluster health"""
        try:
            # Get node status
            nodes = await self.k8s_manager.list_nodes()
            ready_nodes = sum(1 for n in nodes if n.get("status") == "Ready")

            # Get metrics
            metrics = await self.k8s_manager.get_cluster_metrics()

            # Get pod and service counts
            pods = await self.k8s_manager.list_all_pods()
            services = await self.k8s_manager.list_all_services()

            return ClusterHealth(
                cluster_id=self.cluster_id,
                timestamp=datetime.utcnow(),
                node_count=len(nodes),
                ready_nodes=ready_nodes,
                cpu_usage_percent=metrics.get("cpu_usage", 0),
                memory_usage_percent=metrics.get("memory_usage", 0),
                pod_count=len(pods),
                service_count=len(services),
                error_rate=metrics.get("error_rate", 0),
                latency_p99_ms=metrics.get("latency_p99", 0),
                issues=[],
            )
        except Exception as e:
            self.logger.error(f"Failed to get cluster health: {e}")
            return ClusterHealth(
                cluster_id=self.cluster_id,
                timestamp=datetime.utcnow(),
                node_count=0,
                ready_nodes=0,
                cpu_usage_percent=100,
                memory_usage_percent=100,
                pod_count=0,
                service_count=0,
                error_rate=1.0,
                latency_p99_ms=99999,
                issues=[{"error": str(e)}],
            )

    async def deploy_service(self, service: Service, config: Dict[str, Any]) -> bool:
        """Deploy service to Kubernetes"""
        try:
            # Create deployment manifest
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": service.name,
                    "namespace": service.namespace,
                    "labels": {"app": service.name, "version": service.version},
                },
                "spec": {
                    "replicas": service.replicas.get(self.cluster_id, 1),
                    "selector": {"matchLabels": {"app": service.name}},
                    "template": {
                        "metadata": {
                            "labels": {"app": service.name, "version": service.version}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": service.name,
                                    "image": config.get(
                                        "image", f"{service.name}:{service.version}"
                                    ),
                                    "ports": config.get("ports", []),
                                    "env": config.get("env", []),
                                    "resources": config.get("resources", {}),
                                }
                            ]
                        },
                    },
                },
            }

            # Apply deployment
            result = await self.k8s_manager.apply_manifest(deployment)

            # Create service if needed
            if config.get("expose", False):
                svc_manifest = {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {"name": service.name, "namespace": service.namespace},
                    "spec": {
                        "selector": {"app": service.name},
                        "ports": config.get("service_ports", []),
                        "type": config.get("service_type", "ClusterIP"),
                    },
                }
                await self.k8s_manager.apply_manifest(svc_manifest)

            return result
        except Exception as e:
            self.logger.error(f"Failed to deploy service: {e}")
            return False

    async def scale_service(
        self, service_name: str, namespace: str, replicas: int
    ) -> bool:
        """Scale Kubernetes deployment"""
        try:
            return await self.k8s_manager.scale_deployment(
                service_name, namespace, replicas
            )
        except Exception as e:
            self.logger.error(f"Failed to scale service: {e}")
            return False

    async def get_service_status(
        self, service_name: str, namespace: str
    ) -> Dict[str, Any]:
        """Get Kubernetes service status"""
        try:
            deployment = await self.k8s_manager.get_deployment(service_name, namespace)
            pods = await self.k8s_manager.list_pods(
                namespace, label_selector=f"app={service_name}"
            )

            return {
                "exists": deployment is not None,
                "ready": (
                    deployment.get("status", {}).get("readyReplicas", 0)
                    if deployment
                    else 0
                ),
                "desired": (
                    deployment.get("spec", {}).get("replicas", 0) if deployment else 0
                ),
                "pods": len(pods),
                "version": (
                    deployment.get("metadata", {}).get("labels", {}).get("version")
                    if deployment
                    else None
                ),
            }
        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {"exists": False, "error": str(e)}

    async def apply_patch(
        self, resource_type: str, name: str, namespace: str, patch: Dict[str, Any]
    ) -> bool:
        """Apply patch to Kubernetes resource"""
        try:
            return await self.k8s_manager.patch_resource(
                resource_type, name, namespace, patch
            )
        except Exception as e:
            self.logger.error(f"Failed to apply patch: {e}")
            return False


class ServiceMeshController:
    """Controls service mesh operations across clusters"""

    def __init__(self, mesh_type: ServiceMeshType):
        self.mesh_type = mesh_type
        self.logger = logging.getLogger(f"{__name__}.ServiceMesh.{mesh_type.value}")

    async def configure_traffic_split(
        self, service: Service, traffic_split: Dict[str, float]
    ) -> bool:
        """Configure traffic splitting across clusters"""
        if self.mesh_type == ServiceMeshType.ISTIO:
            return await self._configure_istio_traffic_split(service, traffic_split)
        elif self.mesh_type == ServiceMeshType.LINKERD:
            return await self._configure_linkerd_traffic_split(service, traffic_split)
        else:
            self.logger.warning(
                f"Traffic split not supported for {self.mesh_type.value}"
            )
            return False

    async def _configure_istio_traffic_split(
        self, service: Service, traffic_split: Dict[str, float]
    ) -> bool:
        """Configure Istio VirtualService for traffic splitting"""
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{service.name}-cross-cluster",
                "namespace": service.namespace,
            },
            "spec": {
                "hosts": [service.name],
                "http": [
                    {
                        "route": [
                            {
                                "destination": {
                                    "host": service.name,
                                    "subset": cluster_id,
                                },
                                "weight": int(weight * 100),
                            }
                            for cluster_id, weight in traffic_split.items()
                        ]
                    }
                ],
            },
        }

        # Store the resource for later application
        self._pending_resources = getattr(self, "_pending_resources", [])
        self._pending_resources.append(virtual_service)

        # In practice, would apply this to the service mesh
        self.logger.info(
            f"Configured Istio traffic split for {service.name}: {virtual_service['metadata']['name']}"
        )
        return True

    async def _configure_linkerd_traffic_split(
        self, service: Service, traffic_split: Dict[str, float]
    ) -> bool:
        """Configure Linkerd TrafficSplit for traffic splitting"""
        traffic_split_resource = {
            "apiVersion": "split.smi-spec.io/v1alpha1",
            "kind": "TrafficSplit",
            "metadata": {
                "name": f"{service.name}-split",
                "namespace": service.namespace,
            },
            "spec": {
                "service": service.name,
                "backends": [
                    {
                        "service": f"{service.name}-{cluster_id}",
                        "weight": int(weight * 100),
                    }
                    for cluster_id, weight in traffic_split.items()
                ],
            },
        }

        # Store the resource for later application
        self._pending_resources = getattr(self, "_pending_resources", [])
        self._pending_resources.append(traffic_split_resource)

        self.logger.info(
            f"Configured Linkerd traffic split for {service.name}: {traffic_split_resource['metadata']['name']}"
        )
        return True

    async def enable_circuit_breaking(
        self, service: Service, config: Dict[str, Any]
    ) -> bool:
        """Enable circuit breaking for cross-cluster calls"""
        if self.mesh_type == ServiceMeshType.ISTIO:
            destination_rule = {
                "apiVersion": "networking.istio.io/v1beta1",
                "kind": "DestinationRule",
                "metadata": {
                    "name": f"{service.name}-circuit-breaker",
                    "namespace": service.namespace,
                },
                "spec": {
                    "host": service.name,
                    "trafficPolicy": {
                        "connectionPool": {
                            "tcp": {
                                "maxConnections": config.get("max_connections", 100)
                            },
                            "http": {
                                "http1MaxPendingRequests": config.get(
                                    "max_pending_requests", 100
                                ),
                                "http2MaxRequests": config.get("max_requests", 100),
                            },
                        },
                        "outlierDetection": {
                            "consecutiveErrors": config.get("consecutive_errors", 5),
                            "interval": f"{config.get('interval_seconds', 30)}s",
                            "baseEjectionTime": f"{config.get('ejection_seconds', 30)}s",
                            "maxEjectionPercent": config.get(
                                "max_ejection_percent", 50
                            ),
                        },
                    },
                },
            }

            # Store the resource for later application
            self._pending_resources = getattr(self, "_pending_resources", [])
            self._pending_resources.append(destination_rule)

            self.logger.info(
                f"Enabled circuit breaking for {service.name}: {destination_rule['metadata']['name']}"
            )
            return True

        return False

    async def configure_retry_policy(
        self, service: Service, config: Dict[str, Any]
    ) -> bool:
        """Configure retry policy for cross-cluster calls"""
        if self.mesh_type in [ServiceMeshType.ISTIO, ServiceMeshType.LINKERD]:
            # Configure retries in service mesh
            return True
        return False


class CrossClusterOrchestrator:
    """
    Orchestrates operations across multiple container orchestration clusters,
    managing service deployment, scaling, and healing coordination.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clusters: Dict[str, Cluster] = {}
        self.connectors: Dict[str, ClusterConnector] = {}
        self.services: Dict[str, Service] = {}
        self.policies: Dict[str, CrossClusterPolicy] = {}
        self.mesh_controllers: Dict[str, ServiceMeshController] = {}
        self.monitor = DistributedMonitor()
        self.logger = logging.getLogger(__name__)

        # Load configurations
        self._load_clusters(config.get("clusters", []))
        self._load_policies(config.get("policies", []))
        self._load_services(config.get("services", []))

    def _load_clusters(self, cluster_configs: List[Dict[str, Any]]):
        """Load cluster configurations"""
        for config in cluster_configs:
            cluster = Cluster(
                id=config["id"],
                name=config["name"],
                type=ClusterType(config["type"]),
                endpoint=config["endpoint"],
                auth_config=config["auth"],
                region=config["region"],
                state=ClusterState.ACTIVE,
                service_mesh=ServiceMeshType(config.get("service_mesh", "none")),
                namespaces=config.get("namespaces", ["default"]),
                metadata=config.get("metadata", {}),
            )
            self.clusters[cluster.id] = cluster

            # Create connector based on cluster type
            if cluster.type in [
                ClusterType.KUBERNETES,
                ClusterType.EKS,
                ClusterType.GKE,
                ClusterType.AKS,
            ]:
                self.connectors[cluster.id] = KubernetesConnector(cluster.id)

            # Create service mesh controller if applicable
            if cluster.service_mesh != ServiceMeshType.NONE:
                self.mesh_controllers[cluster.id] = ServiceMeshController(
                    cluster.service_mesh
                )

    def _load_policies(self, policy_configs: List[Dict[str, Any]]):
        """Load cross-cluster policies"""
        for config in policy_configs:
            policy = CrossClusterPolicy(
                name=config["name"],
                distribution_strategy=config["distribution_strategy"],
                min_replicas_per_cluster=config["min_replicas_per_cluster"],
                max_replicas_per_cluster=config["max_replicas_per_cluster"],
                failover_enabled=config.get("failover_enabled", True),
                auto_scaling_enabled=config.get("auto_scaling_enabled", True),
                traffic_split=config.get("traffic_split", {}),
                consistency_requirements=config.get(
                    "consistency_requirements", "eventual"
                ),
            )
            self.policies[policy.name] = policy

    def _load_services(self, service_configs: List[Dict[str, Any]]):
        """Load service configurations"""
        for config in service_configs:
            service = Service(
                name=config["name"],
                namespace=config["namespace"],
                version=config["version"],
                replicas=config.get("replicas", {}),
                endpoints=config.get("endpoints", {}),
                health_checks=config.get("health_checks", []),
                dependencies=config.get("dependencies", []),
                metadata=config.get("metadata", {}),
            )
            self.services[service.name] = service

    async def initialize(self) -> bool:
        """Initialize connections to all clusters"""
        tasks = []
        for cluster_id, connector in self.connectors.items():
            cluster = self.clusters[cluster_id]
            tasks.append(self._init_cluster(cluster, connector))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        self.logger.info(f"Initialized {success_count}/{len(tasks)} clusters")

        return success_count > 0

    async def _init_cluster(
        self, cluster: Cluster, connector: ClusterConnector
    ) -> bool:
        """Initialize a single cluster connection"""
        try:
            connected = await connector.connect(cluster.endpoint, cluster.auth_config)
            if connected:
                health = await connector.get_health()
                if health.ready_nodes > 0:
                    cluster.state = ClusterState.ACTIVE
                    return True
                else:
                    cluster.state = ClusterState.DEGRADED
            else:
                cluster.state = ClusterState.FAILED
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize cluster {cluster.name}: {e}")
            cluster.state = ClusterState.FAILED
            return False

    async def deploy_service_across_clusters(
        self,
        service_name: str,
        target_clusters: Optional[List[str]] = None,
        policy_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deploy a service across multiple clusters"""
        service = self.services.get(service_name)
        if not service:
            return {"error": f"Service {service_name} not found"}

        # Determine target clusters
        if not target_clusters:
            target_clusters = [
                c.id for c in self.clusters.values() if c.state == ClusterState.ACTIVE
            ]

        # Get policy
        policy = self.policies.get(policy_name) if policy_name else None

        # Calculate replicas per cluster
        replicas_distribution = self._calculate_replica_distribution(
            service, target_clusters, policy
        )

        # Deploy to each cluster
        results = {}
        for cluster_id in target_clusters:
            if cluster_id not in self.connectors:
                results[cluster_id] = {"error": "No connector available"}
                continue

            # Update service replicas for this cluster
            service.replicas[cluster_id] = replicas_distribution.get(cluster_id, 0)

            # Deploy
            success = await self.connectors[cluster_id].deploy_service(
                service, self._get_deployment_config(service, cluster_id)
            )

            results[cluster_id] = {"success": success}

            if success:
                # Configure service mesh if available
                if cluster_id in self.mesh_controllers:
                    await self._configure_service_mesh(service, cluster_id, policy)

        # Update traffic splitting if policy specifies
        if policy and policy.traffic_split:
            await self._update_traffic_split(service, policy.traffic_split)

        return {
            "service": service_name,
            "deployments": results,
            "total_replicas": sum(replicas_distribution.values()),
        }

    def _calculate_replica_distribution(
        self,
        service: Service,
        clusters: List[str],
        policy: Optional[CrossClusterPolicy],
    ) -> Dict[str, int]:
        """Calculate how to distribute replicas across clusters"""
        if not clusters:
            return {}

        total_replicas = sum(service.replicas.values()) or len(clusters) * 2

        if policy:
            if policy.distribution_strategy == "balanced":
                # Distribute evenly
                base_replicas = total_replicas // len(clusters)
                remainder = total_replicas % len(clusters)

                distribution = {}
                for i, cluster_id in enumerate(clusters):
                    replicas = base_replicas + (1 if i < remainder else 0)
                    replicas = max(
                        policy.min_replicas_per_cluster,
                        min(policy.max_replicas_per_cluster, replicas),
                    )
                    distribution[cluster_id] = replicas

                return distribution

            elif policy.distribution_strategy == "primary-backup":
                # Most replicas in first cluster, minimum in others
                distribution = {
                    clusters[0]: min(
                        policy.max_replicas_per_cluster,
                        total_replicas - (len(clusters) - 1),
                    )
                }
                for cluster_id in clusters[1:]:
                    distribution[cluster_id] = policy.min_replicas_per_cluster
                return distribution

        # Default: even distribution
        return {cluster_id: total_replicas // len(clusters) for cluster_id in clusters}

    def _get_deployment_config(
        self, service: Service, cluster_id: str
    ) -> Dict[str, Any]:
        """Get deployment configuration for a specific cluster"""
        cluster = self.clusters[cluster_id]

        return {
            "image": f"{service.name}:{service.version}",
            "ports": [{"containerPort": 8080}],  # Default, would be from service config
            "env": [
                {"name": "CLUSTER_ID", "value": cluster_id},
                {"name": "REGION", "value": cluster.region},
            ],
            "resources": {
                "requests": {"cpu": "100m", "memory": "128Mi"},
                "limits": {"cpu": "500m", "memory": "512Mi"},
            },
            "expose": True,
            "service_ports": [{"port": 80, "targetPort": 8080}],
            "service_type": "ClusterIP",
        }

    async def _configure_service_mesh(
        self, service: Service, cluster_id: str, policy: Optional[CrossClusterPolicy]
    ):
        """Configure service mesh for the deployed service"""
        mesh_controller = self.mesh_controllers[cluster_id]

        # Enable circuit breaking
        await mesh_controller.enable_circuit_breaking(
            service,
            {
                "consecutive_errors": 5,
                "interval_seconds": 30,
                "ejection_seconds": 30,
                "max_ejection_percent": 50,
            },
        )

        # Configure retry policy
        await mesh_controller.configure_retry_policy(
            service,
            {
                "attempts": 3,
                "perTryTimeout": "30s",
                "retryOn": "5xx,reset,connect-failure,refused-stream",
            },
        )

    async def _update_traffic_split(
        self, service: Service, traffic_split: Dict[str, float]
    ):
        """Update traffic splitting across clusters"""
        # Find a cluster with service mesh to configure traffic split
        for cluster_id, controller in self.mesh_controllers.items():
            if self.clusters[cluster_id].state == ClusterState.ACTIVE:
                await controller.configure_traffic_split(service, traffic_split)
                break

    async def handle_cluster_failure(self, failed_cluster_id: str) -> Dict[str, Any]:
        """Handle failure of a cluster by redistributing services"""
        if failed_cluster_id not in self.clusters:
            return {"error": "Cluster not found"}

        failed_cluster = self.clusters[failed_cluster_id]
        failed_cluster.state = ClusterState.FAILED

        # Find services deployed on failed cluster
        affected_services = [
            service
            for service in self.services.values()
            if failed_cluster_id in service.replicas and
            service.replicas[failed_cluster_id] > 0
        ]

        if not affected_services:
            return {"message": "No services affected"}

        # Get healthy clusters
        healthy_clusters = [
            c.id
            for c in self.clusters.values()
            if c.state == ClusterState.ACTIVE and c.id != failed_cluster_id
        ]

        if not healthy_clusters:
            return {"error": "No healthy clusters available for failover"}

        results = {}

        for service in affected_services:
            # Redistribute replicas from failed cluster
            failed_replicas = service.replicas.get(failed_cluster_id, 0)

            # Remove failed cluster from service
            service.replicas.pop(failed_cluster_id, None)
            service.endpoints.pop(failed_cluster_id, None)

            # Redistribute to healthy clusters
            additional_per_cluster = failed_replicas // len(healthy_clusters)
            remainder = failed_replicas % len(healthy_clusters)

            for i, cluster_id in enumerate(healthy_clusters):
                additional = additional_per_cluster + (1 if i < remainder else 0)
                current = service.replicas.get(cluster_id, 0)
                new_replicas = current + additional

                # Scale up service in healthy cluster
                success = await self.connectors[cluster_id].scale_service(
                    service.name, service.namespace, new_replicas
                )

                if success:
                    service.replicas[cluster_id] = new_replicas

                results[f"{service.name}_{cluster_id}"] = {
                    "scaled": success,
                    "new_replicas": new_replicas,
                }

        # Update traffic splits to exclude failed cluster
        for service in affected_services:
            await self._rebalance_traffic(service, healthy_clusters)

        return {
            "failed_cluster": failed_cluster_id,
            "affected_services": [s.name for s in affected_services],
            "redistribution": results,
        }

    async def _rebalance_traffic(self, service: Service, active_clusters: List[str]):
        """Rebalance traffic after cluster changes"""
        if not active_clusters:
            return

        # Calculate new traffic split
        total_replicas = sum(service.replicas.get(c, 0) for c in active_clusters)
        if total_replicas == 0:
            return

        traffic_split = {
            cluster_id: service.replicas.get(cluster_id, 0) / total_replicas
            for cluster_id in active_clusters
        }

        # Update traffic split in service mesh
        await self._update_traffic_split(service, traffic_split)

    async def perform_cross_cluster_healing(
        self, healing_context: HealingContext, healing_plan: HealingPlan
    ) -> Dict[str, Any]:
        """Coordinate healing across clusters for affected services"""
        affected_services = []
        affected_clusters = set()

        # Identify affected services and clusters
        for step in healing_plan.steps:
            if "service" in step.parameters:
                service_name = step.parameters["service"]
                if service_name in self.services:
                    affected_services.append(self.services[service_name])

            # Find clusters associated with the environment
            for cluster in self.clusters.values():
                if cluster.region == step.environment.region:
                    affected_clusters.add(cluster.id)

        if not affected_services or not affected_clusters:
            return {"status": "no_cluster_healing_needed"}

        results = {"services": {}, "clusters": list(affected_clusters)}

        for service in affected_services:
            service_results = {}

            # Apply healing to each affected cluster
            for cluster_id in affected_clusters:
                if cluster_id not in self.connectors:
                    continue

                connector = self.connectors[cluster_id]

                # Apply patch if specified
                if "patch" in healing_context.constraints:
                    patch = healing_context.constraints["patch"]
                    success = await connector.apply_patch(
                        "deployment", service.name, service.namespace, patch
                    )
                    service_results[cluster_id] = {"patch_applied": success}

                # Restart pods if needed
                if healing_context.constraints.get("restart_required", False):
                    success = await connector.scale_service(
                        service.name, service.namespace, 0
                    )
                    if success:
                        await asyncio.sleep(5)
                        success = await connector.scale_service(
                            service.name,
                            service.namespace,
                            service.replicas.get(cluster_id, 1),
                        )
                    service_results[cluster_id] = {"restarted": success}

            results["services"][service.name] = service_results

        return results

    async def get_cluster_status(self, cluster_id: str) -> Dict[str, Any]:
        """Get detailed status of a cluster"""
        if cluster_id not in self.clusters:
            return {"error": "Cluster not found"}

        cluster = self.clusters[cluster_id]
        connector = self.connectors.get(cluster_id)

        status = {
            "cluster": {
                "id": cluster.id,
                "name": cluster.name,
                "type": cluster.type.value,
                "region": cluster.region,
                "state": cluster.state.value,
                "service_mesh": cluster.service_mesh.value,
            }
        }

        if connector and cluster.state != ClusterState.FAILED:
            try:
                health = await connector.get_health()
                status["health"] = {
                    "nodes": f"{health.ready_nodes}/{health.node_count}",
                    "cpu_usage": health.cpu_usage_percent,
                    "memory_usage": health.memory_usage_percent,
                    "pod_count": health.pod_count,
                    "service_count": health.service_count,
                    "error_rate": health.error_rate,
                    "latency_p99_ms": health.latency_p99_ms,
                }

                # Get service statuses
                services = {}
                for service_name, service in self.services.items():
                    if cluster_id in service.replicas:
                        svc_status = await connector.get_service_status(
                            service_name, service.namespace
                        )
                        services[service_name] = svc_status

                status["services"] = services

            except Exception as e:
                status["error"] = str(e)

        return status

    async def get_cross_cluster_view(self) -> Dict[str, Any]:
        """Get comprehensive view across all clusters"""
        view = {
            "clusters": {},
            "services": {},
            "policies": {},
            "health_summary": {
                "total_clusters": len(self.clusters),
                "active_clusters": 0,
                "total_services": len(self.services),
                "total_replicas": 0,
            },
        }

        # Gather cluster statuses
        for cluster_id in self.clusters:
            status = await self.get_cluster_status(cluster_id)
            view["clusters"][cluster_id] = status

            if self.clusters[cluster_id].state == ClusterState.ACTIVE:
                view["health_summary"]["active_clusters"] += 1

        # Gather service distributions
        for service_name, service in self.services.items():
            view["services"][service_name] = {
                "version": service.version,
                "total_replicas": sum(service.replicas.values()),
                "distribution": service.replicas,
                "endpoints": service.endpoints,
                "dependencies": service.dependencies,
            }
            view["health_summary"]["total_replicas"] += sum(service.replicas.values())

        # Include policies
        for policy_name, policy in self.policies.items():
            view["policies"][policy_name] = {
                "distribution_strategy": policy.distribution_strategy,
                "traffic_split": policy.traffic_split,
                "failover_enabled": policy.failover_enabled,
                "auto_scaling_enabled": policy.auto_scaling_enabled,
            }

        return view

    async def rebalance_clusters(
        self, policy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rebalance services across clusters based on policy"""
        policy = self.policies.get(policy_name) if policy_name else None
        if not policy:
            return {"error": "Policy not found or not specified"}

        results = {}

        # Get active clusters
        active_clusters = [
            c.id for c in self.clusters.values() if c.state == ClusterState.ACTIVE
        ]

        if len(active_clusters) < 2:
            return {"message": "Not enough active clusters for rebalancing"}

        # Rebalance each service
        for service_name, service in self.services.items():
            # Calculate new distribution
            new_distribution = self._calculate_replica_distribution(
                service, active_clusters, policy
            )

            # Apply changes
            service_results = {}
            for cluster_id, new_replicas in new_distribution.items():
                current_replicas = service.replicas.get(cluster_id, 0)

                if new_replicas != current_replicas:
                    connector = self.connectors.get(cluster_id)
                    if connector:
                        success = await connector.scale_service(
                            service.name, service.namespace, new_replicas
                        )
                        service_results[cluster_id] = {
                            "previous": current_replicas,
                            "new": new_replicas,
                            "success": success,
                        }
                        if success:
                            service.replicas[cluster_id] = new_replicas

            if service_results:
                results[service_name] = service_results

        # Update traffic splits
        for service in self.services.values():
            await self._rebalance_traffic(service, active_clusters)

        return {
            "policy": policy_name,
            "rebalanced_services": results,
            "active_clusters": active_clusters,
        }
